local function launch(w, h, args)
    return cuda.kernel({args=args, block={16,16}, grid={math.ceil(w/16),math.ceil(h/16)}})
end

function raster(inputs, outputs)
    local h=inputs["SceneColor"].shape[1]
    local w=inputs["SceneColor"].shape[2]
    outputs["Depth"]=cuda.image(w,h,cuda.float)
    outputs["Optical"]=cuda.image(w,h,cuda.float2)
    local layers=inputs["opticalLayers"].value
    outputs["Layers"]=cuda.image(w,h*layers,cuda.float2)
    return launch(w,h,{
        cuda.int(w),cuda.int(h),cuda.int(inputs["tileSize"]),
        cuda.float(inputs["tanHalfFov"]),cuda.float(inputs["near"]),cuda.float(inputs["far"]),
        cuda.float(inputs["radiusScale"]),cuda.int(inputs["opticalLayers"]),cuda.float(inputs["layerWidth"]),
        cuda.uint64(inputs["particles"]),cuda.uint64(inputs["dye"]),
        cuda.uint64(inputs["offsets"]),cuda.uint64(inputs["indices"]),
        cuda.uint64(inputs["kernels"]),
        cuda.TextureObject(inputs["SceneDepth"]),cuda.SurfaceObject(outputs["Depth"]),
        cuda.SurfaceObject(outputs["Optical"]),
        cuda.SurfaceObject(outputs["Layers"]),
    })
end

local function filterLaunch(inputs, outputs)
    local h=inputs["Depth"].shape[1]
    local w=inputs["Depth"].shape[2]
    outputs["Depth"]=cuda.image(w,h,cuda.float)
    return launch(w,h,{
        cuda.int(w),cuda.int(h),cuda.int(inputs["axis"]),cuda.float(inputs["radius"]),
        cuda.float(inputs["tanHalfFov"]),cuda.TextureObject(inputs["Depth"]),
        cuda.SurfaceObject(outputs["Depth"]),
    })
end

function narrow(inputs, outputs)
    return filterLaunch(inputs, outputs)
end

function curvature(inputs, outputs)
    return filterLaunch(inputs, outputs)
end

function expand_depth(inputs, outputs)
    local h=inputs["Depth"].shape[1]
    local w=inputs["Depth"].shape[2]
    outputs["Depth"]=cuda.image(w,h,cuda.float)
    return launch(w,h,{cuda.int(w),cuda.int(h),cuda.TextureObject(inputs["Depth"]),
                       cuda.SurfaceObject(outputs["Depth"])})
end

function close_depth(inputs, outputs)
    local h=inputs["Original"].shape[1]
    local w=inputs["Original"].shape[2]
    outputs["Depth"]=cuda.image(w,h,cuda.float)
    return launch(w,h,{cuda.int(w),cuda.int(h),cuda.float(inputs["radius"]),
                       cuda.TextureObject(inputs["Original"]),cuda.TextureObject(inputs["Expanded"]),
                       cuda.SurfaceObject(outputs["Depth"])})
end

function smooth_optical(inputs, outputs)
    local h=inputs["Depth"].shape[1]
    local w=inputs["Depth"].shape[2]
    local opticalHeight=inputs["Optical"].shape[1]
    outputs["Optical"]=cuda.image(w,opticalHeight,cuda.float2)
    return launch(w,opticalHeight,{
        cuda.int(w),cuda.int(h),cuda.int(opticalHeight/h),cuda.int(inputs["axis"]),cuda.float(inputs["radius"]),
        cuda.float(inputs["tanHalfFov"]),cuda.TextureObject(inputs["Depth"]),
        cuda.TextureObject(inputs["Optical"]),cuda.SurfaceObject(outputs["Optical"]),
    })
end

function shade(inputs, outputs)
    local h=inputs["SceneColor"].shape[1]
    local w=inputs["SceneColor"].shape[2]
    outputs["Color"]=cuda.image(w,h,cuda.uchar4)
    local args={
        cuda.int(w),cuda.int(h),cuda.float(inputs["tanHalfFov"]),
        cuda.float(inputs["absorption"]),cuda.float(inputs["ior"]),
        cuda.float(inputs["radius"]),cuda.float(inputs["roughness"]),
        cuda.int(inputs["colorModel"]),cuda.int(inputs["debug"]),
    }
    -- Pass components explicitly. The installed SPG build's float3 kernel
    -- arguments produce frame-varying components despite stable USD inputs.
    for _,name in ipairs({"right","up","forward","keyDirection","keyColor","domeColor"}) do
        for component=1,3 do
            args[#args+1]=cuda.float(inputs[name].value[component])
        end
    end
    args[#args+1]=cuda.int(inputs["mediumModel"])
    args[#args+1]=cuda.float(inputs["mediumScale"])
    args[#args+1]=cuda.int(inputs["opticalLayers"])
    for _,name in ipairs({"absorptionA","scatteringA","absorptionB","scatteringB"}) do
        for component=1,3 do
            args[#args+1]=cuda.float(inputs[name].value[component])
        end
    end
    local resources={
        cuda.TextureObject(inputs["Depth"]),cuda.TextureObject(inputs["Optical"]),
        cuda.TextureObject(inputs["SceneColor"]),cuda.TextureObject(inputs["SceneDepth"]),
        cuda.TextureObject(inputs["Layers"]),
        cuda.SurfaceObject(outputs["Color"]),
    }
    for _,resource in ipairs(resources) do args[#args+1]=resource end
    return launch(w,h,args)
end
