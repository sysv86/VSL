[[vk::binding(0,0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(1,0)]] RWStructuredBuffer<uint> result;
[shader("compute")]
[numthreads(1,1,1)]
void main(uint3 gid : SV_DispatchThreadID) {
    RayDesc ray;
    ray.Origin = float3(float(gid.x)*4.0,0,0);
    ray.Direction = float3(0,0,1);
    ray.TMin = 0.001;
    ray.TMax = 100;
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(tlas,RAY_FLAG_NONE,255,ray);
    while(q.Proceed()) {
        if(q.CandidateType()==CANDIDATE_NON_OPAQUE_TRIANGLE) q.CommitNonOpaqueTriangleHit();
    }
    uint value = 0;
    if(q.CommittedStatus()==COMMITTED_TRIANGLE_HIT) {
        value=uint(q.CommittedRayT()*100)+q.CommittedPrimitiveIndex()+1;
    }
    result[gid.x]=value;
}
