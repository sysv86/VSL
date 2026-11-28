[[vk::binding(0,0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(1,0)]] RWStructuredBuffer<uint> result;
struct Payload { uint value; };
[shader("raygeneration")]
void rgen() {
    uint3 gid = DispatchRaysIndex();
    RayDesc ray;
    ray.Origin = float3(float(gid.x)*4.0,0,0);
    ray.Direction = float3(0,0,1);
    ray.TMin = 0.001;
    ray.TMax = 100;
    Payload p;
    p.value=999;
    TraceRay(tlas,RAY_FLAG_NONE,255,0,0,0,ray,p);
    result[gid.x]=p.value;
}
[shader("closesthit")]
void rchit(inout Payload p, BuiltInTriangleIntersectionAttributes attr) {
    p.value=uint(RayTCurrent()*100)+1;
}
[shader("miss")]
void rmiss(inout Payload p) { p.value=0; }
