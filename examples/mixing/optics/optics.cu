// The launcher prepends the existing SSFR kernels and helpers.
extern "C" __global__ void rear_depth(
    int w, int h, int tileSize, float fov, float nearPlane, float farPlane, float binWidth,
    unsigned long long positions, unsigned long long offsets, unsigned long long indices,
    unsigned long long kernels, cudaTextureObject_t rawFront, cudaTextureObject_t scene,
    cudaSurfaceObject_t output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float frontZ = depthAt(rawFront,x,y,w,h);
    float exitZ = 0.0f;
    if (frontZ > 0.0f) {
        const float4* particles = (const float4*)positions;
        const int* starts = (const int*)offsets;
        const int* ids = (const int*)indices;
        const float* transforms = (const float*)kernels;
        float3 ray = unit3(viewRay(x,y,w,h,fov));
        float origin = frontZ / ray.z;
        float far = fminf(farPlane / ray.z, sceneDistance(scene,x,y,farPlane / ray.z));
        int tile = (y/tileSize)*((w+tileSize-1)/tileSize) + x/tileSize;
        unsigned long long occupied = 0;
        for (int j=starts[tile]; j<starts[tile+1]; ++j) {
            int i = ids[j]; float enter, leave;
            if (!particleChord(ray,particles[i],transforms+9*i,nearPlane/ray.z,far,enter,leave)) continue;
            int lo = max(0, (int)floorf((enter-origin)/binWidth));
            int hi = min(63, (int)floorf((leave-origin)/binWidth));
            if (lo <= hi) occupied |= ((~0ull) << lo) & ((~0ull) >> (63-hi));
        }
        if (occupied) {
            int firstGap = occupied == ~0ull ? 64 : __ffsll((long long)~occupied)-1;
            float lastConnected = origin + firstGap * binWidth;
            float exit = origin;
            for (int j=starts[tile]; j<starts[tile+1]; ++j) {
                int i=ids[j]; float enter, leave;
                if (particleChord(ray,particles[i],transforms+9*i,nearPlane/ray.z,far,enter,leave)
                    && enter < lastConnected) exit = fmaxf(exit,leave);
            }
            exitZ = exit * ray.z;
        }
    }
    surf2Dwrite(exitZ,output,x*sizeof(float),y);
}

extern "C" __global__ void smooth_rear(
    int w,int h,float radius,cudaTextureObject_t front,cudaTextureObject_t rear,cudaSurfaceObject_t output)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h) return;
    float z=depthAt(front,x,y,w,h), b=depthAt(rear,x,y,w,h);
    float sum=0, norm=0;
    if(z>0 && b>z) {
        for(int dy=-2;dy<=2;++dy) for(int dx=-2;dx<=2;++dx) {
            float f=depthAt(front,x+dx,y+dy,w,h);
            float r=depthAt(rear,x+dx,y+dy,w,h);
            if(f<=0 || r<=f || fabsf(f-z)>3*radius || fabsf(r-b)>5*radius) continue;
            float df=(f-z)/radius, db=(r-b)/(2*radius);
            float weight=expf(-(dx*dx+dy*dy)/5.0f - 0.5f*(df*df+db*db));
            sum+=weight*r; norm+=weight;
        }
    }
    surf2Dwrite(norm>0?fmaxf(z+0.0005f,sum/norm):0.0f,output,x*sizeof(float),y);
}

__device__ float4 mixPanorama(float4 a,float4 b,float t) {
    float3 rgb=mix3(make_float3(a.x,a.y,a.z),make_float3(b.x,b.y,b.z),t);
    return make_float4(rgb.x,rgb.y,rgb.z,a.w+(b.w-a.w)*t);
}
__device__ float4 samplePanorama(const float4* map,int w,int h,float3 dir,float longitudeShift) {
    // This axis convention is verified against HdrColor on the Z-up OVRTX stage.
    float u=atan2f(dir.y,-dir.x)*(1.0f/6.28318530718f)+0.5f+longitudeShift;
    u-=floorf(u);
    float v=acosf(fminf(1.0f,fmaxf(-1.0f,dir.z)))*(1.0f/3.14159265359f);
    float px=u*w-0.5f, py=fminf(h-1.0f,fmaxf(0.0f,v*h-0.5f));
    int x=(int)floorf(px), y=(int)floorf(py);
    int a=(x+w)%w,b=(x+1+w)%w, yy=min(y+1,h-1);
    float4 p=map[y*w+a],q=map[y*w+b],r=map[yy*w+a],s=map[yy*w+b];
    return mixPanorama(mixPanorama(p,q,px-x),mixPanorama(r,s,px-x),py-y);
}
__device__ float3 hdrSample(const float4* map,int w,int h,float3 dir,float longitudeShift) {
    float4 value=samplePanorama(map,w,h,dir,longitudeShift);
    return make_float3(value.x,value.y,value.z);
}
__device__ float4 sampleFilteredPanorama(const float4* map,int w,int h,float3 dir,float longitudeShift,float roughness) {
    return mixPanorama(samplePanorama(map,w,h,dir,longitudeShift),
                       samplePanorama(map+w*h,w,h,dir,longitudeShift),sat(roughness*5.0f));
}
__device__ float3 keyPanoramaDirection(float3 ray,float3 key) {
    // The disk is centered on -X in a light-local panorama. Transforming the
    // ray keeps light aim independent of the DomeLight's stage rotation.
    float3 axis=fabsf(key.z)<0.999f?make_float3(0,0,1):make_float3(0,1,0);
    float3 right=unit3(cross3(axis,key));
    float3 up=cross3(key,right);
    return make_float3(-dot3(ray,key),dot3(ray,right),dot3(ray,up));
}
__device__ float3 toWorld(float3 d,float3 right,float3 up,float3 forward) {
    return add3(add3(mul3(right,d.x),mul3(up,d.y)),mul3(forward,d.z));
}
__device__ float3 hdrPixel(cudaTextureObject_t input,int x,int y) {
    float4 p=tex2D<float4>(input,x,y); return make_float3(p.x,p.y,p.z);
}
__device__ float3 displayCurve(float3 c) {
    float3 out;
    out.x=sat((c.x*(2.51f*c.x+0.03f))/(c.x*(2.43f*c.x+0.59f)+0.14f));
    out.y=sat((c.y*(2.51f*c.y+0.03f))/(c.y*(2.43f*c.y+0.59f)+0.14f));
    out.z=sat((c.z*(2.51f*c.z+0.03f))/(c.z*(2.43f*c.z+0.59f)+0.14f));
    return out;
}
__device__ bool projectView(float3 p,int w,int h,float fov,int& x,int& y) {
    if(p.z<=0.001f) return false;
    float scale=h/(2*fov*p.z);
    x=(int)floorf(0.5f*w+p.x*scale);
    y=(int)floorf(0.5f*h-p.y*scale);
    return x>=0&&x<w&&y>=0&&y<h;
}
__device__ float3 reflectDir(float3 incident,float3 normal) {
    return sub3(incident,mul3(normal,2*dot3(incident,normal)));
}
__device__ bool dielectric(float3 incident,float3 normal,float n1,float n2,float3& transmitted,float& F) {
    float ci=-dot3(incident,normal);
    if(ci<0) { normal=mul3(normal,-1); ci=-ci; }
    ci=sat(ci);
    float eta=n1/n2, k=1-eta*eta*(1-ci*ci);
    if(k<0) { F=1; return false; }
    float ct=sqrtf(k);
    float rs=(n1*ci-n2*ct)/fmaxf(1e-6f,n1*ci+n2*ct);
    float rp=(n2*ci-n1*ct)/fmaxf(1e-6f,n2*ci+n1*ct);
    F=0.5f*(rs*rs+rp*rp);
    transmitted=unit3(add3(mul3(incident,eta),mul3(normal,eta*ci-ct)));
    return true;
}


// 1: opaque scene; 2: rear interface; 4: front interface.
__device__ int crossing(float z,float sceneZ,float frontZ,float rearZ,bool inside) {
    if(sceneZ <= z && z-sceneZ<0.05f && (!inside||rearZ<=0||sceneZ<=rearZ+0.002f)) return 1;
    if(inside && rearZ>0 && z>=rearZ && z-rearZ<0.06f) return 2;
    if(inside && frontZ>0 && z<frontZ-0.001f && frontZ-z<0.04f) return 4;
    return 0;
}
__device__ int screenEvent(float3 p,int w,int h,float fov,
    cudaTextureObject_t scene,cudaTextureObject_t front,cudaTextureObject_t rear,
    bool inside,int& x,int& y,float& rearZ)
{
    if(!projectView(p,w,h,fov,x,y)) return -1;
    float sceneZ=sceneDistance(scene,x,y,1.0e20f)*unit3(viewRay(x,y,w,h,fov)).z;
    rearZ=inside?depthAt(rear,x,y,w,h):0.0f;
    float frontZ=inside?depthAt(front,x,y,w,h):0.0f;
    return crossing(p.z,sceneZ,frontZ,rearZ,inside);
}

__device__ int traceScreen(float3 start,float3 ray,int w,int h,float fov,
    cudaTextureObject_t scene,cudaTextureObject_t front,cudaTextureObject_t rear,
    bool inside,int& sx,int& sy,float3& position)
{
    float previous=0.0f;
    for(int i=0; i<64;++i) {
        float t=previous+(i<20?0.006f:0.017f);
        float3 p=add3(start,mul3(ray,t));
        int x,y; float rearZ;
        int event=screenEvent(p,w,h,fov,scene,front,rear,inside,x,y,rearZ);
        if(event<0) return 0;
        if(event>0) {
            float lo=previous,hi=t; sx=x;sy=y;
            for(int j=0;j<5;++j) {
                float middle=0.5f*(lo+hi);
                float3 q=add3(start,mul3(ray,middle));
                int ax,ay; float ignored;
                int hit=screenEvent(q,w,h,fov,scene,front,rear,inside,ax,ay,ignored);
                if(hit>0) {hi=middle;sx=ax;sy=ay;event=hit;}
                else lo=middle;
            }
            position=add3(start,mul3(ray,hi));
            return event;
        }
        if(inside&&i>2&&rearZ<=0.0f) return 0;
        previous=t;
    }
    return 0;
}

__device__ void coffeeMilkCoefficients(float c,float3& extinction,float3& albedo) {
    mediumCoefficients(c,make_float3(14.4f,36.0f,80.0f),make_float3(.15f,.25f,.5f),
                       make_float3(.1f,.1f,.1f),make_float3(80,80,80),extinction,albedo);
}

__device__ float smoothUnit(float x) {
    x=sat(x);
    return x*x*(3.0f-2.0f*x);
}

__device__ float2 layerAtPoint(
    int x,int y,int w,int h,float pointZ,float rayScale,float layerWidth,int layerCount,
    cudaTextureObject_t front,cudaTextureObject_t layers)
{
    float z=depthAt(front,x,y,w,h);
    if(z<=0.0f) return make_float2(0,0);
    float depth=(pointZ-z)*rayScale;
    float coverage=smoothUnit((depth+0.004f)/0.008f);
    if(coverage<=0.0f) return make_float2(0,0);
    float level=fminf(layerCount-1.0f,fmaxf(0.0f,depth/layerWidth-0.5f));
    int lo=(int)floorf(level), hi=min(lo+1,layerCount-1);
    float2 a=tex2D<float2>(layers,x,y+lo*h), b=tex2D<float2>(layers,x,y+hi*h);
    float fraction=level-lo;
    return make_float2(coverage*(a.x+(b.x-a.x)*fraction),coverage*(a.y+(b.y-a.y)*fraction));
}

__device__ float3 incidentTransmission(float3 start,float3 towardLight,int w,int h,float fov,
    float layerWidth,int layerCount,cudaTextureObject_t front,cudaTextureObject_t layers)
{
    const float step=0.020f;
    float3 opticalDepth=make_float3(0,0,0);
    for(int sample=0;sample<6;++sample) {
        float3 p=add3(start,mul3(towardLight,(sample+0.5f)*step));
        if(p.z<=0.001f) break;
        float scale=h/(2.0f*fov*p.z);
        float px=0.5f*w+p.x*scale-0.5f, py=0.5f*h-p.y*scale-0.5f;
        int x=(int)floorf(px),y=(int)floorf(py);
        if(x<0||x+1>=w||y<0||y+1>=h) break;
        float rayScale=sqrtf(dot3(p,p))/p.z;
        float2 pack=make_float2(0,0);
        for(int dy=0;dy<2;++dy) for(int dx=0;dx<2;++dx) {
            float weight=(dx?px-x:1.0f-(px-x))*(dy?py-y:1.0f-(py-y));
            float2 layer=layerAtPoint(x+dx,y+dy,w,h,p.z,rayScale,layerWidth,layerCount,front,layers);
            pack.x+=weight*layer.x;
            pack.y+=weight*layer.y;
        }
        float3 extinction,unused;
        coffeeMilkCoefficients(sat(pack.y/fmaxf(pack.x,1e-10f)),extinction,unused);
        opticalDepth=add3(opticalDepth,mul3(extinction,step*sat(pack.x/layerWidth)));
    }
    return mediumTransmission(opticalDepth,1.0f,1.0f);
}

extern "C" __global__ void hdri_shade(
    int w,int h,float fov,float radius,float roughness,float ior,int mode,
    float r0,float r1,float r2,float u0,float u1,float u2,float f0,float f1,float f2,
    float lx,float ly,float lz,float keyPower,float keyAngle,float hdriScale,float exposure,
    float ambR,float ambG,float ambB,unsigned long long hdriAddress,int envW,int envH,int layerCount,
    float envLongitudeShift,int keyFromPanorama,
    int directionalVolume,float layerWidth,float cameraHeight,float volumeKeyScale,
    cudaTextureObject_t front,cudaTextureObject_t back,cudaTextureObject_t optical,cudaTextureObject_t layers,
    cudaTextureObject_t hdr,cudaTextureObject_t scene,cudaSurfaceObject_t output)
{
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=w||y>=h) return;
    float3 background=hdrPixel(hdr,x,y), result=mode==6?make_float3(0,0,0):background;
    float z=depthAt(front,x,y,w,h), b=depthAt(back,x,y,w,h);
    float2 optics=tex2D<float2>(optical,x,y);
    float3 ray=unit3(viewRay(x,y,w,h,fov));
    if(z>0.0f&&optics.x>0.0f&&z/ray.z<sceneDistance(scene,x,y,1.0e20f)) {
        float3 right=make_float3(r0,r1,r2),up=make_float3(u0,u1,u2),forward=make_float3(f0,f1,f2);
        float3 key=unit3(make_float3(lx,ly,lz));
        const float4* panorama=(const float4*)hdriAddress;
        float variance;
        float3 n=surfaceNormal(front,x,y,w,h,fov,z,radius,variance);
        float3 inside; float fresnel;
        dielectric(ray,n,1.0f,ior,inside,fresnel);
        float3 viewWorld=toWorld(mul3(ray,-1),right,up,forward);
        float3 worldNormal=toWorld(n,right,up,forward);
        float3 worldReflection=toWorld(reflectDir(ray,n),right,up,forward);
        float4 dome=sampleFilteredPanorama(panorama,envW,envH,worldReflection,envLongitudeShift,roughness);
        float3 domeReflection=mul3(make_float3(dome.x,dome.y,dome.z),hdriScale*fresnel);
        float spec;
        if(keyFromPanorama) {
            float3 keyRay=keyPanoramaDirection(worldReflection,key);
            spec=sampleFilteredPanorama(panorama,envW,envH,keyRay,0.0f,roughness).w*fresnel*keyPower;
        } else {
            float f=(ior-1.0f)/(ior+1.0f);
            spec=ggxSpecular(worldNormal,viewWorld,key,f*f,roughness,
                            variance+keyAngle*keyAngle*0.25f)*keyPower;
        }
        float3 reflection=add3(domeReflection,make_float3(spec,spec,spec));
        if(mode>=3 && mode<=5) {
            float3 pass=mode==4?domeReflection:mode==5?make_float3(spec,spec,spec):reflection;
            float3 v=displayCurve(mul3(pass,exposure));
            uchar4 color=make_uchar4(linearToByte(v.x),linearToByte(v.y),linearToByte(v.z),255);
            surf2Dwrite(color,output,x*sizeof(uchar4),y);
            return;
        }

        float3 entry=mul3(ray,z/ray.z), position, outRay=inside;
        int sx=x,sy=y,path=0;
        float exitFresnel=0.0f,pathRatio=1.0f;
        float3 through;
        if(mode==1) {
            float3 q=add3(entry,mul3(inside,fminf(optics.x,0.25f)));
            int ax,ay;
            if(projectView(q,w,h,fov,ax,ay) && sceneDistance(scene,ax,ay,1e20f)*unit3(viewRay(ax,ay,w,h,fov)).z>=z) {
                sx=ax; sy=ay;
            }
            through=hdrPixel(hdr,sx,sy);
        } else {
            path=traceScreen(entry,inside,w,h,fov,scene,front,back,true,sx,sy,position);
            if(path&&b>z+0.003f) {
                float3 displacement=sub3(position,entry);
                pathRatio=fminf(1.8f,fmaxf(0.65f,sqrtf(dot3(displacement,displacement))/((b-z)/ray.z)));
            }
            if(path==1) through=hdrPixel(hdr,sx,sy);
            else if(path==2||path==4) {
                cudaTextureObject_t surface=path==2?back:front;
                float bz=depthAt(surface,sx,sy,w,h), exitVar;
                float3 nr=surfaceNormal(surface,sx,sy,w,h,fov,bz,radius,exitVar);
                if(dielectric(inside,nr,ior,1.0f,outRay,exitFresnel)) {
                    int ax,ay; float3 unused;
                    int hit=traceScreen(add3(position,mul3(outRay,0.004f)),outRay,w,h,fov,scene,front,back,false,ax,ay,unused);
                    through=hit==1?hdrPixel(hdr,ax,ay):
                        mul3(hdrSample(panorama,envW,envH,toWorld(outRay,right,up,forward),envLongitudeShift),hdriScale);
                } else {
                    float3 bounce=reflectDir(inside,nr), unused;
                    int ax,ay;
                    int hit=traceScreen(add3(position,mul3(bounce,0.005f)),bounce,w,h,fov,scene,front,back,true,ax,ay,unused);
                    through=make_float3(0,0,0); exitFresnel=0.0f;
                    if(hit==1) through=hdrPixel(hdr,ax,ay);
                    if(hit==2||hit==4) {
                        cudaTextureObject_t second=hit==2?back:front;
                        float secondZ=depthAt(second,ax,ay,w,h), secondVar;
                        float3 secondN=surfaceNormal(second,ax,ay,w,h,fov,secondZ,radius,secondVar);
                        if(dielectric(bounce,secondN,ior,1.0f,outRay,exitFresnel)) {
                            through=mul3(hdrSample(panorama,envW,envH,toWorld(outRay,right,up,forward),envLongitudeShift),hdriScale);
                            int exitX,exitY; float3 noPosition;
                            if(traceScreen(add3(unused,mul3(outRay,0.004f)),outRay,w,h,fov,
                                           scene,front,back,false,exitX,exitY,noPosition)==1)
                                through=hdrPixel(hdr,exitX,exitY);
                        }
                    }
                    path=3;
                }
            } else {
                through=mul3(hdrSample(panorama,envW,envH,toWorld(inside,right,up,forward),envLongitudeShift),hdriScale);
            }
        }

        float3 ambient=make_float3(ambR,ambG,ambB), towardLight=make_float3(0,0,0);
        float directionalWeight=0.0f, lightFresnel=0.0f;
        if(directionalVolume && key.z>0.0f) {
            float height=cameraHeight+dot3(entry,make_float3(r2,u2,f2));
            float2 first=tex2D<float2>(layers,x,y);
            float2 second=layerCount>1?tex2D<float2>(layers,x,y+h):make_float2(0,0);
            float milk=(first.y+second.y)/fmaxf(first.x+second.x,1e-10f);
            // Preserve the milk jet, including its lower neck near the pool.
            float pool=1.0f-smoothUnit((height-0.185f)/0.05f);
            float mixed=1.0f-smoothUnit((milk-0.84f)/0.14f);
            directionalWeight=0.45f*pool*mixed;
            if(directionalWeight>0.0f) {
                float3 lightInside;
                dielectric(mul3(key,-1),make_float3(0,0,1),1.0f,ior,lightInside,lightFresnel);
                towardLight=make_float3(-dot3(lightInside,right),-dot3(lightInside,up),-dot3(lightInside,forward));
            }
        }
        float3 T=make_float3(1,1,1), S=make_float3(0,0,0);
        for(int l=0;l<layerCount;++l) {
            float2 pack=tex2D<float2>(layers,x,y+l*h);
            float c=sat(pack.y/fmaxf(pack.x,1e-10f));
            float3 ext,albedo;
            coffeeMilkCoefficients(c,ext,albedo);
            float3 m=mediumTransmission(ext,pack.x,pathRatio);
            float3 incident=ambient;
            if(l<3 && directionalWeight>0.0f && pack.x>1e-5f) {
                float3 center=add3(entry,mul3(ray,(l+0.5f)*layerWidth));
                float3 shadow=incidentTransmission(center,towardLight,w,h,fov,layerWidth,layerCount,front,layers);
                float gain=1.6f*volumeKeyScale*(1.0f-lightFresnel);
                float3 direct=make_float3(ambR*shadow.x,ambG*shadow.y,ambB*shadow.z);
                incident=mix3(ambient,mul3(direct,gain),directionalWeight*(l==2?0.65f:1.0f));
            }
            S=add3(S,make_float3(T.x*albedo.x*(1-m.x)*incident.x,
                                 T.y*albedo.y*(1-m.y)*incident.y,
                                 T.z*albedo.z*(1-m.z)*incident.z));
            T=make_float3(T.x*m.x,T.y*m.y,T.z*m.z);
        }
        float3 medium=add3(S,mul3(make_float3(through.x*T.x,through.y*T.y,through.z*T.z),1-exitFresnel));
        result=mix3(background,add3(reflection,mul3(medium,1-fresnel)),sat(optics.x*h/(2*z*fov)));
        if(mode==6) result=mul3(S,1.0f-fresnel);
        if(mode==2) result=path==1?make_float3(.08f,.75f,.25f):(path==2||path==4)?make_float3(.04f,.28f,1.0f)
                            :path==3?make_float3(.65f,.05f,.9f):make_float3(1,.4f,.06f);
    }
    float3 display=displayCurve(mul3(result,exposure));
    uchar4 color=make_uchar4(linearToByte(display.x),linearToByte(display.y),linearToByte(display.z),255);
    surf2Dwrite(color,output,x*sizeof(uchar4),y);
}
