// Screen-space fluid rendering, positive eye-space depth, top-left image origin.
// Narrow-range filtering follows Truong and Yuksel, I3D 2018, equations 1-9.
// CUDA buffers are owned by FluidBuffers and stay alive until RTX finishes.

__device__ float3 add3(float3 a, float3 b) { return make_float3(a.x+b.x,a.y+b.y,a.z+b.z); }
__device__ float3 sub3(float3 a, float3 b) { return make_float3(a.x-b.x,a.y-b.y,a.z-b.z); }
__device__ float3 mul3(float3 a, float b) { return make_float3(a.x*b,a.y*b,a.z*b); }
__device__ float dot3(float3 a, float3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__device__ float3 cross3(float3 a, float3 b) {
    return make_float3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);
}
__device__ float3 unit3(float3 a) { return mul3(a,rsqrtf(fmaxf(dot3(a,a),1.0e-20f))); }
__device__ float sat(float x) { return fminf(1.0f,fmaxf(0.0f,x)); }
__device__ float3 mix3(float3 a, float3 b, float t) { return add3(mul3(a,1-t),mul3(b,t)); }
__device__ float3 transformKernel(const float* matrix,float3 value) {
    return make_float3(matrix[0]*value.x+matrix[1]*value.y+matrix[2]*value.z,
                       matrix[3]*value.x+matrix[4]*value.y+matrix[5]*value.z,
                       matrix[6]*value.x+matrix[7]*value.y+matrix[8]*value.z);
}
__device__ float3 viewRay(int x,int y,int w,int h,float tanHalfFov) {
    float scale=2.0f*tanHalfFov/h;
    return make_float3((x+0.5f-0.5f*w)*scale,(0.5f*h-y-0.5f)*scale,1.0f);
}
__device__ float depthAt(cudaTextureObject_t image,int x,int y,int w,int h) {
    if(x<0||x>=w||y<0||y>=h) return 0.0f;
    return tex2D<float>(image,x,y);
}
__device__ float sceneDistance(cudaTextureObject_t image,int x,int y,float farDistance) {
    float d=tex2D<float>(image,x,y);
    return isfinite(d)&&d>0.0f?d:farDistance;
}

__device__ bool particleChord(float3 ray,float4 particle,const float* kernel,
                              float nearDistance,float limit,float& front,float& back) {
    float3 center=make_float3(particle.x,particle.y,particle.z);
    float3 kernelRay=transformKernel(kernel,ray);
    float3 kernelCenter=transformKernel(kernel,center);
    float rayLength2=dot3(kernelRay,kernelRay);
    float closestParameter=dot3(kernelRay,kernelCenter)/rayLength2;
    float3 residual=sub3(kernelCenter,mul3(kernelRay,closestParameter));
    float radial2=1.0f-dot3(residual,residual);
    if(radial2<=0.0f) return false;
    float halfChord=sqrtf(radial2/rayLength2);
    front=fmaxf(closestParameter-halfChord,nearDistance);
    back=fminf(closestParameter+halfChord,limit);
    return back>front;
}

extern "C" __global__ void raster(
    int w,int h,int tileSize,float tanHalfFov,float nearPlane,float farPlane,float radiusScale,
    int opticalLayers,float layerWidth,
    unsigned long long particleAddress,unsigned long long dyeAddress,
    unsigned long long offsetAddress,unsigned long long indexAddress,unsigned long long kernelAddress,
    cudaTextureObject_t sceneDepth,cudaSurfaceObject_t depth,cudaSurfaceObject_t optical,
    cudaSurfaceObject_t layeredOptical)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h) return;
    const float4* particles=(const float4*)particleAddress;
    const float* dye=(const float*)dyeAddress;
    const int* offsets=(const int*)offsetAddress;
    const int* indices=(const int*)indexAddress;
    const float* kernels=(const float*)kernelAddress;
    float3 ray=unit3(viewRay(x,y,w,h,tanHalfFov));
    float limit=fminf(farPlane/ray.z,sceneDistance(sceneDepth,x,y,farPlane/ray.z));
    float closest=farPlane, thickness=0.0f, dyeThickness=0.0f;
    int tilesX=(w+tileSize-1)/tileSize, tile=(y/tileSize)*tilesX+x/tileSize;
    // Particle volume is spacing^3, the input radius is spacing/2. Correct the
    // inflated sphere volume so radius tuning does not change bulk opacity.
    float volumeWeight=6.0f/(3.14159265359f*radiusScale*radiusScale*radiusScale);
    for(int entry=offsets[tile];entry<offsets[tile+1];++entry) {
        int i=indices[entry];
        float front,back;
        if(!particleChord(ray,particles[i],kernels+9*i,nearPlane/ray.z,limit,front,back)) continue;
        closest=fminf(closest,front*ray.z);
        float chord=(back-front)*volumeWeight;
        thickness+=chord;
        dyeThickness+=chord*sat(dye[i]);
    }
    surf2Dwrite(thickness>0.0f?closest:0.0f,depth,x*sizeof(float),y);
    surf2Dwrite(make_float2(thickness,dyeThickness),optical,x*sizeof(float2),y);
    if(opticalLayers>1) {
        // Front-to-back slabs retain ordering of separate jets and the
        // pool. The final slab includes the entire remaining ray segment.
        // Keep this capacity in sync with media.MAX_MEDIUM_LAYERS.
        float layerThickness[16]={0}, layerDye[16]={0};
        float origin=closest/ray.z;
        for(int entry=offsets[tile];entry<offsets[tile+1]&&thickness>0.0f;++entry) {
            int i=indices[entry];
            float front,back;
            if(!particleChord(ray,particles[i],kernels+9*i,nearPlane/ray.z,limit,front,back)) continue;
            int first=min(opticalLayers-1,max(0,(int)floorf((front-origin)/layerWidth)));
            int last=min(opticalLayers-1,max(0,(int)floorf((back-origin)/layerWidth)));
            for(int layer=first;layer<=last;++layer) {
                float start=fmaxf(front,origin+layer*layerWidth);
                float end=layer==opticalLayers-1?back:fminf(back,origin+(layer+1)*layerWidth);
                float chord=fmaxf(0.0f,end-start)*volumeWeight;
                layerThickness[layer]+=chord;
                layerDye[layer]+=chord*sat(dye[i]);
            }
        }
        for(int layer=0;layer<opticalLayers;++layer) {
            surf2Dwrite(make_float2(layerThickness[layer],layerDye[layer]),layeredOptical,
                        x*sizeof(float2),y+layer*h);
        }
    } else {
        surf2Dwrite(make_float2(thickness,dyeThickness),layeredOptical,x*sizeof(float2),y);
    }
}

// Positive depth reverses the inequalities in the paper's negative-Z notation.
// Process opposing pixels together to keep the filter support symmetric.
__device__ void narrowPair(float a,float b,float center,float delta,float mu,float weight,
                           float& low,float& high,float& sum,float& weights) {
    if((a>0.0f&&a<center-high)||(b>0.0f&&b<center-high)) return;
    bool acceptA=a>0.0f&&a<=center+low;
    bool acceptB=b>0.0f&&b<=center+low;
    float va=acceptA?a:center+mu, vb=acceptB?b:center+mu;
    sum+=weight*(va+vb);
    weights+=2.0f*weight;
    if(acceptA) { low=fmaxf(low,a-center+delta); high=fmaxf(high,center-a+delta); }
    if(acceptB) { low=fmaxf(low,b-center+delta); high=fmaxf(high,center-b+delta); }
}

extern "C" __global__ void expand_depth(int w,int h,cudaTextureObject_t source,cudaSurfaceObject_t target)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h) return;
    float result=0.0f;
    for(int dy=-2;dy<=2;++dy) {
        for(int dx=-2;dx<=2;++dx) {
            if(dx*dx+dy*dy>4) continue;
            float z=depthAt(source,x+dx,y+dy,w,h);
            if(z>0.0f) result=result>0.0f?fminf(result,z):z;
        }
    }
    surf2Dwrite(result,target,x*sizeof(float),y);
}

extern "C" __global__ void close_depth(int w,int h,float radius,cudaTextureObject_t original,
                                      cudaTextureObject_t expanded,cudaSurfaceObject_t target)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h) return;
    float result=depthAt(original,x,y,w,h);
    if(result<=0.0f) {
        float center=depthAt(expanded,x,y,w,h);
        bool covered=center>0.0f;
        float sum=0.0f,weights=0.0f;
        for(int dy=-2;dy<=2;++dy) {
            for(int dx=-2;dx<=2;++dx) {
                if(dx*dx+dy*dy>4||x+dx<0||x+dx>=w||y+dy<0||y+dy>=h) continue;
                float z=depthAt(expanded,x+dx,y+dy,w,h);
                covered=covered&&z>0.0f&&fabsf(z-center)<3.0f*radius;
                float sample=depthAt(original,x+dx,y+dy,w,h);
                if(sample>0.0f&&fabsf(sample-center)<3.0f*radius) {
                    float weight=expf(-0.5f*(dx*dx+dy*dy));
                    sum+=weight*sample; weights+=weight;
                }
            }
        }
        if(covered&&weights>0.0f) result=sum/weights;
    }
    surf2Dwrite(result,target,x*sizeof(float),y);
}

extern "C" __global__ void narrow(int w,int h,int axis,float radius,float tanHalfFov,
                                   cudaTextureObject_t source,cudaSurfaceObject_t target)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h) return;
    float z=depthAt(source,x,y,w,h), result=0.0f;
    if(z>0.0f) {
        // Keep the world-space footprint continuous as particles approach the
        // camera. Rounding sigma makes the whole normal field pop at thresholds.
        float sigma=fminf(32.0f,fmaxf(1.0f,0.9f*radius*h/(2.0f*z*tanHalfFov)));
        float low=10.0f*radius, high=low, sum=z, weights=1.0f;
        if(axis<2) {
            int support=(int)ceilf(3.0f*sigma);
            for(int d=1;d<=support;++d) {
                int dx=axis==0?d:0, dy=axis==1?d:0;
                float a=depthAt(source,x+dx,y+dy,w,h), b=depthAt(source,x-dx,y-dy,w,h);
                float tail=sat((3.0f*sigma-d)/(0.5f*sigma));
                float weight=expf(-d*d/(2.0f*sigma*sigma))*tail*tail*(3.0f-2.0f*tail);
                narrowPair(a,b,z,10.0f*radius,radius,weight,
                           low,high,sum,weights);
            }
        } else {
            // A 5x5 clean-up after the separable passes, nearest ring first.
            for(int ring=1;ring<=2;++ring) {
                for(int dy=0;dy<=ring;++dy) {
                    for(int dx=-ring;dx<=ring;++dx) {
                        if((dy==0&&dx<=0)||max(abs(dx),dy)!=ring) continue;
                        float a=depthAt(source,x+dx,y+dy,w,h), b=depthAt(source,x-dx,y-dy,w,h);
                        narrowPair(a,b,z,10.0f*radius,radius,expf(-(dx*dx+dy*dy)/2.0f),
                                   low,high,sum,weights);
                    }
                }
            }
        }
        result=sum/weights;
    }
    surf2Dwrite(result,target,x*sizeof(float),y);
}

extern "C" __global__ void smooth_optical(int w,int h,int layers,int axis,float radius,float tanHalfFov,
    cudaTextureObject_t depth,cudaTextureObject_t source,cudaSurfaceObject_t target)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h*layers) return;
    int layer=y/h;
    y%=h;
    float z=depthAt(depth,x,y,w,h);
    float2 result=make_float2(0.0f,0.0f);
    if(z>0.0f) {
        float sigma=fminf(32.0f,fmaxf(1.0f,radius*h/(2.0f*z*tanHalfFov)));
        int support=(int)ceilf(3.0f*sigma);
        float weights=0.0f;
        for(int d=-support;d<=support;++d) {
            int sx=x+(axis==0?d:0), sy=y+(axis==1?d:0);
            if(sx<0||sx>=w||sy<0||sy>=h) continue;
            float neighbor=depthAt(depth,sx,sy,w,h);
            // Empty pixels taper coverage, distinct surface layers are rejected.
            if(neighbor>0.0f&&fabsf(neighbor-z)>3.0f*radius) continue;
            float tail=sat((3.0f*sigma-abs(d))/(0.5f*sigma));
            float weight=expf(-d*d/(2.0f*sigma*sigma))*tail*tail*(3.0f-2.0f*tail);
            if(neighbor>0.0f) {
                float difference=(neighbor-z)/radius;
                weight*=expf(-0.5f*difference*difference);
            }
            float2 sample=tex2D<float2>(source,sx,sy+layer*h);
            result.x+=weight*sample.x;
            result.y+=weight*sample.y;
            weights+=weight;
        }
        result.x/=fmaxf(weights,1.0e-10f);
        result.y/=fmaxf(weights,1.0e-10f);
    }
    surf2Dwrite(result,target,x*sizeof(float2),y+layer*h);
}

extern "C" __global__ void curvature(int w,int h,int axis,float radius,float tanHalfFov,
                                      cudaTextureObject_t source,cudaSurfaceObject_t target)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h) return;
    float z=depthAt(source,x,y,w,h), result=z;
    if(z>0.0f) {
        float l=depthAt(source,x-1,y,w,h), r=depthAt(source,x+1,y,w,h);
        float u=depthAt(source,x,y-1,w,h), d=depthAt(source,x,y+1,w,h);
        float ul=depthAt(source,x-1,y-1,w,h), ur=depthAt(source,x+1,y-1,w,h);
        float dl=depthAt(source,x-1,y+1,w,h), dr=depthAt(source,x+1,y+1,w,h);
        // Homogeneous boundary conditions at silhouettes and depth jumps.
        if(l<=0||r<=0||u<=0||d<=0||ul<=0||ur<=0||dl<=0||dr<=0||
           fmaxf(fmaxf(fabsf(l-z),fabsf(r-z)),fmaxf(fabsf(u-z),fabsf(d-z)))>10*radius) {
            surf2Dwrite(result,target,x*sizeof(float),y); return;
        }
        // Equations 4-8 of van der Laan et al. with Cx=Cy for square pixels.
        // Evaluate -H for positive eye depth. The adaptive explicit time step
        // is bounded by the local diffusion CFL, reducing to 0.2*Laplacian.
        float c=2.0f*tanHalfFov/h;
        float zx=0.5f*(r-l), zy=0.5f*(d-u);
        float zxx=r+l-2*z, zyy=d+u-2*z, zxy=0.25f*(dr-dl-ur+ul);
        float D=c*c*(zx*zx+zy*zy+c*c*z*z);
        float Dx=2*c*c*(zx*zxx+zy*zxy+c*c*z*zx);
        float Dy=2*c*c*(zx*zxy+zy*zyy+c*c*z*zy);
        float Ex=0.5f*zx*Dx-zxx*D, Ey=0.5f*zy*Dy-zyy*D;
        float H=0.5f*c*(Ex+Ey)/fmaxf(D*sqrtf(D),1.0e-20f);
        float dt=0.4f*c*z;
        float change=fminf(0.1f*radius,fmaxf(-0.1f*radius,-dt*H));
        result=fmaxf(1.0e-6f,z+change);
    }
    surf2Dwrite(result,target,x*sizeof(float),y);
}

__device__ float3 surfaceNormal(cudaTextureObject_t depth,int x,int y,int w,int h,float fov,
                                float z,float radius,float& variance) {
    // Weighted plane fit with an intercept, including asymmetric silhouette
    // neighborhoods. This avoids switching between two one-sided derivatives.
    float sw=0,sx=0,sy=0,sxx=0,sxy=0,syy=0,sz=0,sxz=0,syz=0,szz=0;
    for(int dy=-2;dy<=2;++dy) {
        for(int dx=-2;dx<=2;++dx) {
            float neighbor=depthAt(depth,x+dx,y+dy,w,h);
            if(neighbor<=0.0f||fabsf(neighbor-z)>3.0f*radius) continue;
            float dz=neighbor-z;
            float weight=expf(-(dx*dx+dy*dy)/4.0f-0.5f*dz*dz/(radius*radius));
            sw+=weight; sx+=weight*dx; sy+=weight*dy;
            sxx+=weight*dx*dx; sxy+=weight*dx*dy; syy+=weight*dy*dy;
            sz+=weight*dz; sxz+=weight*dx*dz; syz+=weight*dy*dz; szz+=weight*dz*dz;
        }
    }
    float xx=sxx-sx*sx/sw, xy=sxy-sx*sy/sw, yy=syy-sy*sy/sw;
    float xz=sxz-sx*sz/sw, yz=syz-sy*sz/sw;
    float determinant=xx*yy-xy*xy;
    float zx=0.0f, zy=0.0f;
    if(determinant>1.0e-8f) {
        zx=(xz*yy-yz*xy)/determinant;
        zy=(yz*xx-xz*xy)/determinant;
    }
    float pixelWorld=2.0f*z*fov/h;
    float residual=fmaxf(0.0f,(szz-sz*sz/sw-zx*xz-zy*yz)/sw);
    variance=fminf(0.01f,residual/(4.0f*pixelWorld*pixelWorld+1.0e-12f));
    float3 ray=viewRay(x,y,w,h,fov);
    float3 dx=add3(mul3(ray,zx),make_float3(pixelWorld,0,0));
    float3 dy=add3(mul3(ray,zy),make_float3(0,-pixelWorld,0));
    float3 n=unit3(cross3(dx,dy));
    return dot3(n,ray)>0?mul3(n,-1.0f):n;
}
__device__ float srgbToLinear(float c) { return c<=0.04045f?c/12.92f:powf((c+0.055f)/1.055f,2.4f); }
__device__ unsigned char linearToByte(float c) {
    c=sat(c); c=c<=0.0031308f?12.92f*c:1.055f*powf(c,1.0f/2.4f)-0.055f;
    return (unsigned char)(255.0f*c+0.5f);
}
__device__ float3 readColor(cudaTextureObject_t image,int x,int y) {
    uchar4 c=tex2D<uchar4>(image,x,y);
    return make_float3(srgbToLinear(c.x/255.0f),srgbToLinear(c.y/255.0f),srgbToLinear(c.z/255.0f));
}
__device__ float3 readColorBilinear(cudaTextureObject_t image,float x,float y,int w,int h) {
    x=fminf(w-1.0f,fmaxf(0.0f,x)); y=fminf(h-1.0f,fmaxf(0.0f,y));
    int ix=(int)floorf(x), iy=(int)floorf(y);
    float3 a=mix3(readColor(image,ix,iy),readColor(image,min(ix+1,w-1),iy),x-ix);
    float3 b=mix3(readColor(image,ix,min(iy+1,h-1)),
                  readColor(image,min(ix+1,w-1),min(iy+1,h-1)),x-ix);
    return mix3(a,b,y-iy);
}
__device__ float pigment(float y,float b,float concentration,int model) {
    if(model==0) return y+(b-y)*concentration;
    float ky=(1-y)*(1-y)/(2*y), kb=(1-b)*(1-b)/(2*b), k=ky+(kb-ky)*concentration;
    return 1.0f/(1.0f+k+sqrtf(k*(k+2.0f)));
}
__device__ float3 environment(float3 r,float3 domeColor) {
    float sky=sat(0.5f+0.5f*r.z);
    return mul3(domeColor,0.2f+0.8f*sky);
}
__device__ float ggxSpecular(float3 n,float3 v,float3 light,float f0,float roughness,float variance) {
    float noV=fmaxf(dot3(n,v),1.0e-4f), noL=fmaxf(dot3(n,light),0.0f);
    float3 halfVector=unit3(add3(v,light));
    float noH=sat(dot3(n,halfVector)), loH=sat(dot3(light,halfVector));
    // Broaden unresolved glints with the local fit residual. This lower-bounds
    // the highlight footprint as curvature/noise becomes subpixel.
    float alpha=roughness*roughness;
    float alpha2=fminf(1.0f,alpha*alpha+variance);
    float denominator=noH*noH*(alpha2-1.0f)+1.0f;
    float distribution=alpha2/(3.14159265359f*denominator*denominator);
    float visibility=0.5f/fmaxf(noL*sqrtf(noV*noV*(1.0f-alpha2)+alpha2)
                              +noV*sqrtf(noL*noL*(1.0f-alpha2)+alpha2),1.0e-6f);
    float fresnel=f0>0.0f?f0+(1.0f-f0)*powf(1.0f-loH,5.0f):0.0f;
    return distribution*visibility*fresnel*noL;
}

__device__ float3 mediumTransmission(float3 extinction,float thickness,float scale)
{
    return make_float3(expf(-fminf(80.0f,extinction.x*thickness*scale)),
                       expf(-fminf(80.0f,extinction.y*thickness*scale)),
                       expf(-fminf(80.0f,extinction.z*thickness*scale)));
}

__device__ void mediumCoefficients(float concentration,float3 absorptionA,float3 scatteringA,
    float3 absorptionB,float3 scatteringB,float3& extinction,float3& albedo)
{
    float3 sigmaA=mix3(absorptionA,absorptionB,concentration);
    float3 sigmaS=mix3(scatteringA,scatteringB,concentration);
    extinction=add3(sigmaA,sigmaS);
    albedo=make_float3(sigmaS.x/fmaxf(extinction.x,1.0e-12f),
                       sigmaS.y/fmaxf(extinction.y,1.0e-12f),
                       sigmaS.z/fmaxf(extinction.z,1.0e-12f));
}

extern "C" __global__ void shade(int w,int h,float tanHalfFov,float absorption,float ior,
    float radius,float roughness,int colorModel,int debug,
    float rightX,float rightY,float rightZ,float upX,float upY,float upZ,
    float forwardX,float forwardY,float forwardZ,
    float keyX,float keyY,float keyZ,float keyR,float keyG,float keyB,
    float domeR,float domeG,float domeB,
    int mediumModel,float mediumScale,int opticalLayers,
    float absorptionAR,float absorptionAG,float absorptionAB,
    float scatteringAR,float scatteringAG,float scatteringAB,
    float absorptionBR,float absorptionBG,float absorptionBB,
    float scatteringBR,float scatteringBG,float scatteringBB,
    cudaTextureObject_t depth,cudaTextureObject_t optical,cudaTextureObject_t sceneColor,
    cudaTextureObject_t sceneDepth,cudaTextureObject_t layeredOptical,cudaSurfaceObject_t target)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h) return;
    float3 right=make_float3(rightX,rightY,rightZ), up=make_float3(upX,upY,upZ);
    float3 forward=make_float3(forwardX,forwardY,forwardZ);
    float3 keyDirection=make_float3(keyX,keyY,keyZ), keyColor=make_float3(keyR,keyG,keyB);
    float3 domeColor=make_float3(domeR,domeG,domeB);
    uchar4 original=tex2D<uchar4>(sceneColor,x,y);
    float z=depthAt(depth,x,y,w,h);
    float2 material=tex2D<float2>(optical,x,y);
    float3 ray=unit3(viewRay(x,y,w,h,tanHalfFov));
    if(z<=0||material.x<=0||z/ray.z>=sceneDistance(sceneDepth,x,y,1.0e20f)) {
        surf2Dwrite(original,target,x*sizeof(uchar4),y); return;
    }
    float thickness=material.x, concentration=sat(material.y/fmaxf(thickness,1.0e-12f));
    float normalVariance;
    float3 n=surfaceNormal(depth,x,y,w,h,tanHalfFov,z,radius,normalVariance);
    float3 dyeColor=make_float3(pigment(.95f,.05f,concentration,colorModel),
                               pigment(.85f,.15f,concentration,colorModel),
                               pigment(.05f,.95f,concentration,colorModel));
    float3 transmitted, scattered;
    if(mediumModel) {
        float3 absorptionA=make_float3(absorptionAR,absorptionAG,absorptionAB);
        float3 scatteringA=make_float3(scatteringAR,scatteringAG,scatteringAB);
        float3 absorptionB=make_float3(absorptionBR,absorptionBG,absorptionBB);
        float3 scatteringB=make_float3(scatteringBR,scatteringBG,scatteringBB);
        // A constant incident source integrates analytically through a
        // homogeneous slab. These weights approximate studio illumination.
        float3 incident=add3(mul3(domeColor,2.5f),mul3(keyColor,0.5f));
        transmitted=make_float3(1,1,1);
        scattered=make_float3(0,0,0);
        for(int layer=0;layer<opticalLayers;++layer) {
            float2 sample=tex2D<float2>(layeredOptical,x,y+layer*h);
            float c=sat(sample.y/fmaxf(sample.x,1.0e-12f));
            float3 extinction,albedo;
            mediumCoefficients(c,absorptionA,scatteringA,absorptionB,scatteringB,extinction,albedo);
            float3 attenuation=mediumTransmission(extinction,sample.x,mediumScale);
            scattered=add3(scattered,make_float3(
                transmitted.x*albedo.x*(1-attenuation.x)*incident.x,
                transmitted.y*albedo.y*(1-attenuation.y)*incident.y,
                transmitted.z*albedo.z*(1-attenuation.z)*incident.z));
            transmitted=make_float3(transmitted.x*attenuation.x,transmitted.y*attenuation.y,
                                     transmitted.z*attenuation.z);
        }
        float3 extinction,albedo;
        mediumCoefficients(concentration,absorptionA,scatteringA,absorptionB,scatteringB,extinction,albedo);
        float3 swatch=mediumTransmission(extinction,0.03f,mediumScale);
        dyeColor=make_float3(swatch.x+albedo.x*(1-swatch.x),
                             swatch.y+albedo.y*(1-swatch.y),
                             swatch.z+albedo.z*(1-swatch.z));
    } else {
        transmitted=make_float3(expf(-absorption*(1.02f-dyeColor.x)*thickness),
                                expf(-absorption*(1.02f-dyeColor.y)*thickness),
                                expf(-absorption*(1.02f-dyeColor.z)*thickness));
        scattered=make_float3(.65f*dyeColor.x*(1-transmitted.x),
                              .65f*dyeColor.y*(1-transmitted.y),
                              .65f*dyeColor.z*(1-transmitted.z));
    }
    float cosTheta=sat(-dot3(n,ray)), eta=1.0f/ior;
    float f0=(ior-1)/(ior+1); f0*=f0;
    float fresnel=f0>0.0f?f0+(1-f0)*powf(1-cosTheta,5.0f):0.0f;
    float3 reflection=sub3(ray,mul3(n,2*dot3(ray,n)));
    float3 reflectionWorld=add3(add3(mul3(right,reflection.x),mul3(up,reflection.y)),mul3(forward,reflection.z));
    float3 nWorld=add3(add3(mul3(right,n.x),mul3(up,n.y)),mul3(forward,n.z));
    float3 viewWorld=mul3(add3(add3(mul3(right,ray.x),mul3(up,ray.y)),mul3(forward,ray.z)),-1.0f);
    float specular=ggxSpecular(nWorld,viewWorld,unit3(keyDirection),f0,roughness,normalVariance);
    // The background has already been tone-mapped by RTX. Compress the direct
    // specular into that display-referred range before compositing.
    float3 highlight=mul3(keyColor,specular);
    float luminance=dot3(highlight,make_float3(0.2126f,0.7152f,0.0722f));
    highlight=mul3(highlight,1.0f/(1.0f+luminance));
    float3 reflected=add3(mul3(environment(reflectionWorld,domeColor),fresnel),highlight);
    float3 result;
    if(debug==1) {
        float value=1.0f-expf(-0.6f*z); result=make_float3(value,value,value);
    } else if(debug==2) {
        float value=1.0f-expf(-10.0f*thickness); result=make_float3(value,value,value);
    } else if(debug==3) {
        result=add3(mul3(n,0.5f),make_float3(.5f,.5f,.5f));
    } else if(debug==4) {
        result=dyeColor;
    } else if(debug==5) {
        result=reflected;
    } else if(debug==6) {
        result=transmitted;
    } else if(debug==7) {
        result=scattered;
    } else {
        float discriminant=1.0f-eta*eta*(1-cosTheta*cosTheta);
        float3 refracted=add3(mul3(ray,eta),mul3(n,eta*cosTheta-sqrtf(fmaxf(discriminant,0.0f))));
        float travel=fminf(thickness,0.25f);
        float3 exit=add3(mul3(viewRay(x,y,w,h,tanHalfFov),z),mul3(refracted,travel));
        float rx=fminf(w-1.0f,fmaxf(0.0f,0.5f*w+exit.x/exit.z*h/(2*tanHalfFov)-0.5f));
        float ry=fminf(h-1.0f,fmaxf(0.0f,0.5f*h-exit.y/exit.z*h/(2*tanHalfFov)-0.5f));
        // Reject refraction samples that land on foreground scene geometry.
        float3 rr=unit3(viewRay((int)rx,(int)ry,w,h,tanHalfFov));
        if(sceneDistance(sceneDepth,(int)rx,(int)ry,1.0e20f)*rr.z<z) { rx=x; ry=y; }
        float3 background=readColorBilinear(sceneColor,rx,ry,w,h);
        float3 refractColor=make_float3(background.x*transmitted.x+scattered.x,
                                       background.y*transmitted.y+scattered.y,
                                       background.z*transmitted.z+scattered.z);
        result=add3(mul3(refractColor,1.0f-fresnel),reflected);
        float coverage=sat(thickness/(2.0f*z*tanHalfFov/h));
        result=mix3(readColor(sceneColor,x,y),result,coverage);
    }
    uchar4 color=make_uchar4(linearToByte(result.x),linearToByte(result.y),linearToByte(result.z),255);
    surf2Dwrite(color,target,x*sizeof(uchar4),y);
}
