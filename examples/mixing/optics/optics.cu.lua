local function opticalLaunch(w,h,args)
    return cuda.kernel({args=args,block={16,16},grid={math.ceil(w/16),math.ceil(h/16)}})
end

function rear_depth(inputs, outputs)
    local h=inputs["RawFront"].shape[1]
    local w=inputs["RawFront"].shape[2]
    outputs["Depth"]=cuda.image(w,h,cuda.float)
    return opticalLaunch(w,h,{
        cuda.int(w),cuda.int(h),cuda.int(inputs["tileSize"]),cuda.float(inputs["fov"]),
        cuda.float(inputs["near"]),cuda.float(inputs["far"]),cuda.float(inputs["binWidth"]),
        cuda.uint64(inputs["particles"]),cuda.uint64(inputs["offsets"]),cuda.uint64(inputs["indices"]),
        cuda.uint64(inputs["kernels"]),cuda.TextureObject(inputs["RawFront"]),
        cuda.TextureObject(inputs["SceneDepth"]),cuda.SurfaceObject(outputs["Depth"]),
    })
end

function smooth_rear(inputs, outputs)
    local h=inputs["Front"].shape[1]
    local w=inputs["Front"].shape[2]
    outputs["Depth"]=cuda.image(w,h,cuda.float)
    return opticalLaunch(w,h,{
        cuda.int(w),cuda.int(h),cuda.float(inputs["radius"]),cuda.TextureObject(inputs["Front"]),
        cuda.TextureObject(inputs["Rear"]),cuda.SurfaceObject(outputs["Depth"]),
    })
end

function hdri_shade(inputs, outputs)
    local h=inputs["Front"].shape[1]
    local w=inputs["Front"].shape[2]
    outputs["Color"]=cuda.image(w,h,cuda.uchar4)
    local args={
        cuda.int(w),cuda.int(h),cuda.float(inputs["fov"]),cuda.float(inputs["radius"]),
        cuda.float(inputs["roughness"]),cuda.float(inputs["ior"]),cuda.int(inputs["mode"]),
    }
    for _,name in ipairs({"right","up","forward","keyDirection"}) do
        for c=1,3 do args[#args+1]=cuda.float(inputs[name].value[c]) end
    end
    for _,name in ipairs({"keyPower","keyAngle","envScale","exposure"}) do
        args[#args+1]=cuda.float(inputs[name])
    end
    for c=1,3 do args[#args+1]=cuda.float(inputs["ambient"].value[c]) end
    args[#args+1]=cuda.uint64(inputs["envAddress"])
    for _,name in ipairs({"envWidth","envHeight","layerCount"}) do args[#args+1]=cuda.int(inputs[name]) end
    args[#args+1]=cuda.float(inputs["envLongitudeShift"])
    args[#args+1]=cuda.int(inputs["keyFromPanorama"])
    args[#args+1]=cuda.int(inputs["directionalVolume"])
    for _,name in ipairs({"layerWidth","cameraHeight","volumeKeyScale"}) do
        args[#args+1]=cuda.float(inputs[name])
    end
    for _,name in ipairs({"Front","Rear","Optical","Layers","Hdr","SceneDepth"}) do
        args[#args+1]=cuda.TextureObject(inputs[name])
    end
    args[#args+1]=cuda.SurfaceObject(outputs["Color"])
    return opticalLaunch(w,h,args)
end
