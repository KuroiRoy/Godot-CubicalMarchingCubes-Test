#[compute]
#version 450

// Configuration - must match your CPU implementation
#define CHUNK_SIZE 32
#define CORNERS_PER_CELL 8
#define EDGES_PER_CELL 12
#define FACES_PER_CELL 6
#define MAX_COMPONENTS_PER_CELL 4
#define MAX_SEGMENTS_PER_CELL 12
#define MAX_SEGMENTS_PER_FACE 2

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

// Input Hermite data
struct HermiteData {
    vec3 position;
    vec3 normal;
    float density;
};

layout(set = 0, binding = 0, std430) readonly buffer HermiteBuffer {
    HermiteData hermiteData[];
};

// Output mesh data
layout(set = 0, binding = 1, std430) buffer VertexBuffer {
    vec4 vertices[];
};

layout(set = 0, binding = 2, std430) buffer NormalBuffer {
    vec4 normals[];
};

layout(set = 0, binding = 3, std430) buffer IndexBuffer {
    uint indices[];
};

layout(set = 0, binding = 4, std430) buffer CountBuffer {
    uint vertex_count;
    uint index_count;
};

// Marching Squares case table
struct Case {
    int edgeA;
    int edgeB;
    int edge2A;
    int edge2B;
    int len;
};

const Case marchingSquaresTable[16] = Case[](
    Case(0, 0, 0, 0, 0),   // 0000
    Case(0, 3, 0, 0, 2),   // 0001
    Case(3, 1, 0, 0, 2),   // 0010
    Case(0, 1, 0, 0, 2),   // 0011
    Case(2, 0, 0, 0, 2),   // 0100
    Case(2, 3, 0, 0, 2),   // 0101
    Case(2, 0, 3, 1, 4),   // 0110
    Case(2, 1, 0, 0, 2),   // 0111
    Case(1, 2, 0, 0, 2),   // 1000
    Case(0, 3, 1, 2, 4),   // 1001
    Case(3, 2, 0, 0, 2),   // 1010
    Case(0, 2, 0, 0, 2),   // 1011
    Case(1, 0, 0, 0, 2),   // 1100
    Case(1, 3, 0, 0, 2),   // 1101
    Case(3, 0, 0, 0, 2),   // 1110
    Case(0, 0, 0, 0, 0)    // 1111
);

// Tables
const ivec2 cubeEdgeCornerTable[12] = ivec2[](
    ivec2(2, 0), ivec2(0, 1), ivec2(1, 3), ivec2(3, 2),
    ivec2(6, 4), ivec2(4, 5), ivec2(5, 7), ivec2(7, 6),
    ivec2(0, 4), ivec2(1, 5), ivec2(3, 7), ivec2(2, 6)
);

const ivec4 faceToCubeEdgeTable[6] = ivec4[](
    ivec4(4, 0, 11, 8),   // Left
    ivec4(2, 6, 10, 9),   // Right
    ivec4(11, 10, 7, 3),  // Up
    ivec4(8, 9, 1, 5),     // Down
    ivec4(6, 4, 7, 5),     // Forward
    ivec4(0, 2, 3, 1)      // Back
);

const ivec4 faceToCubeCornerTable[6] = ivec4[](
    ivec4(4, 0, 6, 2),    // Left
    ivec4(1, 5, 3, 7),    // Right
    ivec4(2, 3, 6, 7),    // Up
    ivec4(4, 5, 0, 1),    // Down
    ivec4(5, 4, 7, 6),    // Forward
    ivec4(0, 1, 2, 3)     // Back
);

const ivec2 nextFaceTable[24] = ivec2[](
    ivec2(4, 1), ivec2(5, 0), ivec2(2, 0), ivec2(3, 0), // Left    -> Forward, Back,    Up,      Down
    ivec2(5, 1), ivec2(4, 0), ivec2(2, 1), ivec2(3, 1), // Right   -> Back,    Forward, Up,      Down
    ivec2(0, 2), ivec2(1, 2), ivec2(4, 2), ivec2(5, 2), // Up      -> Left,    Right,   Forward, Back
    ivec2(0, 3), ivec2(1, 3), ivec2(5, 3), ivec2(4, 3), // Down    -> Left,    Right,   Back,    Forward
    ivec2(1, 1), ivec2(0, 0), ivec2(2, 2), ivec2(3, 3), // Forward -> Right,   Left,    Up,      Down
    ivec2(0, 1), ivec2(1, 0), ivec2(2, 3), ivec2(3, 2)  // Back    -> Left,    Right,   Up,      Down 
);

struct Crossing {
    vec3 position;
    vec3 normal;
};

struct Segment {
    int face;
    int startFaceEdge;
    int endFaceEdge;
    int storageIndex;
    bool isUsed;
};

struct QEFData {
    mat3 AtA;       // A^T * A matrix
    vec3 Atb;       // A^T * b vector
    float btb;      // b^T * b scalar
    int numPoints;  // Number of points accumulated
};

void qef_initialize(out QEFData qef) {
    qef.AtA = mat3(0.0);
    qef.Atb = vec3(0.0);
    qef.btb = 0.0;
    qef.numPoints = 0;
}

void qef_add(out QEFData qef, vec3 p, vec3 n) {
    // A is the normal (transposed)
    // b is dot(p, n)
    float b = dot(p, n);
    
    // Accumulate AtA = n * n^T
    qef.AtA[0][0] += n.x * n.x;
    qef.AtA[0][1] += n.x * n.y;
    qef.AtA[0][2] += n.x * n.z;
    qef.AtA[1][0] += n.y * n.x;
    qef.AtA[1][1] += n.y * n.y;
    qef.AtA[1][2] += n.y * n.z;
    qef.AtA[2][0] += n.z * n.x;
    qef.AtA[2][1] += n.z * n.y;
    qef.AtA[2][2] += n.z * n.z;
    
    // Accumulate Atb = n * b
    qef.Atb += n * b;
    
    // Accumulate btb = b * b
    qef.btb += b * b;
    
    qef.numPoints++;
}

// Simplified SVD solver for 3x3 symmetric matrices
vec3 qef_solve(QEFData qef, vec3 massPoint, float svdTolerance, int sweeps) {
    if (qef.numPoints == 0) return massPoint;
    
    mat3 A = qef.AtA;
    vec3 b = qef.Atb;
    
    // Initial guess is the mass point
    vec3 x = massPoint;
    
    // Solve (AtA)x = Atb using iterative refinement
    for (int i = 0; i < sweeps; i++) {
        // Compute residual: r = Atb - AtA*x
        vec3 r = b - A * x;
        
        // Early exit if residual is small enough
        if (dot(r, r) < svdTolerance * svdTolerance) {
            break;
        }
        
        // Update solution: x += r / diag(AtA)
        // (Jacobi preconditioner - simple but effective)
        x += vec3(
            r.x / max(A[0][0], 0.001),
            r.y / max(A[1][1], 0.001),
            r.z / max(A[2][2], 0.001)
        );
    }
    
    // Optional: Project onto the cell if solution is outside
    vec3 cellMin = vec3(gl_GlobalInvocationID.xyz);
    vec3 cellMax = cellMin + vec3(1.0);
    x = clamp(x, cellMin, cellMax);
    
    return x;
}

// Helper functions
vec3 approximateZeroCrossingPosition(vec3 p0, vec3 p1, float v0, float v1) {
    if (abs(v0) < 0.0001) return p0;
    if (abs(v1) < 0.0001) return p1;
    if (abs(v0 - v1) < 0.0001) return p0;
    return p0 - v0 * (p1 - p0) / (v1 - v0);
}

bool closestPointsOnTwoLines(out vec3 closestPointLine1, vec3 linePoint1, vec3 lineVec1, vec3 linePoint2, vec3 lineVec2) {
    closestPointLine1 = vec3(0.0);
    float a = dot(lineVec1, lineVec1);
    float b = dot(lineVec1, lineVec2);
    float e = dot(lineVec2, lineVec2);
    float d = a * e - b * b;

    if (abs(d) > 0.0001) {
        vec3 r = linePoint1 - linePoint2;
        float c = dot(lineVec1, r);
        float f = dot(lineVec2, r);
        float s = (b * f - c * e) / d;
        closestPointLine1 = linePoint1 + lineVec1 * s;
        return true;
    }
    return false;
}

void main() {
    ivec3 cellPos = ivec3(gl_GlobalInvocationID.xyz);
    if (cellPos.x >= CHUNK_SIZE-1 || cellPos.y >= CHUNK_SIZE-1 || cellPos.z >= CHUNK_SIZE-1) {
        return;
    }

    // Get Hermite data for this cell
    HermiteData cornerData[8];
    for (int i = 0; i < 8; i++) {
        ivec3 corner = cellPos + ivec3((i & 1), ((i >> 1) & 1), ((i >> 2) & 1));
        int index = corner.x + corner.y * CHUNK_SIZE + corner.z * CHUNK_SIZE * CHUNK_SIZE;
        cornerData[i] = hermiteData[index];
    }

    // Find edge crossings
    Crossing edgeCrossings[12];
    for (int edge = 0; edge < 12; edge++) {
        ivec2 corners = cubeEdgeCornerTable[edge];
        HermiteData data1 = cornerData[corners.x];
        HermiteData data2 = cornerData[corners.y];
        
        if (data1.density * data2.density > 0.0) continue;
        
        vec3 position = approximateZeroCrossingPosition(
            data1.position, 
            data2.position, 
            data1.density, 
            data2.density
        );
        
        vec3 normal = normalize(data1.normal * abs(data1.density) + data2.normal * abs(data2.density));
        edgeCrossings[edge] = Crossing(position, normal);
    }

    // Face processing
    Segment segmentsByFace[6][2];

    for (int face = 0; face < 6; face++) {
        // Reset segments for this face
        segmentsByFace[face][0] = Segment(-1, -1, -1, 0, false);
        segmentsByFace[face][1] = Segment(-1, -1, -1, 1, false);
        
        // Get marching squares case for this face
        ivec4 faceCorners = faceToCubeCornerTable[face];
        int index = 0;
        
        if (cornerData[faceCorners.x].density <= 0.0) index |= 1;
        if (cornerData[faceCorners.y].density <= 0.0) index |= 2;
        if (cornerData[faceCorners.z].density <= 0.0) index |= 4;
        if (cornerData[faceCorners.w].density <= 0.0) index |= 8;
        
        Case currentCase = marchingSquaresTable[index];
        
        // Create segments for this face
        if (currentCase.len > 0) {
            segmentsByFace[face][0] = Segment(face, currentCase.edgeA, currentCase.edgeB, 0, false);
        }
        
        if (currentCase.len > 2) {
            segmentsByFace[face][1] = Segment(face, currentCase.edge2A, currentCase.edge2B, 1, false);
        }
    }

    // Link segments into components
    Segment segmentsByComponent[4][12];
    int segmentsPerComponent[4] = int[](0, 0, 0, 0);

    for (int component = 0; component < 4; component++) {
        // Find first available segment
        Segment firstSegment = Segment(-1, -1, -1, -1, true);
        for (int face = 0; face < 6; face++) {
            if (segmentsByFace[face][0].face != -1 && !segmentsByFace[face][0].isUsed) {
                firstSegment = segmentsByFace[face][0];
                break;
            }

            if (segmentsByFace[face][1].face != -1 && !segmentsByFace[face][1].isUsed) {
                firstSegment = segmentsByFace[face][1];
                break;
            }
        }
        
        if (firstSegment.face == -1 || firstSegment.isUsed) break; // No more segments to use for components

        // Find segments that link to the current segment
        Segment nextSegment = firstSegment;
        int componentSegmentIndex = 0;
        do {
            // Store the segment in the component and increment the index
            segmentsByComponent[component][componentSegmentIndex] = nextSegment;
            componentSegmentIndex++;

            // Mark the current segment as invalid for other components
            segmentsByFace[nextSegment.face][nextSegment.storageIndex].isUsed = true;

            int nextFace = nextFaceTable[nextSegment.face * 4 + nextSegment.endFaceEdge].x;
            int nextFaceEdge = nextFaceTable[nextSegment.face * 4 + nextSegment.endFaceEdge].y;
            
            Segment candidate = segmentsByFace[nextFace][0];
            if (candidate.face != -1 && !candidate.isUsed && candidate.startFaceEdge == nextFaceEdge) {
                nextSegment = candidate;
                continue;
            }
            candidate = segmentsByFace[nextFace][1];
            if (candidate.face != -1 && !candidate.isUsed && candidate.startFaceEdge == nextFaceEdge) {
                nextSegment = candidate;
                continue;
            }
            break; // Components should be valid every time, no valid segment was found
        }
        while (!(
            nextSegment.face == firstSegment.face &&
            nextSegment.startFaceEdge == firstSegment.startFaceEdge &&
            nextSegment.endFaceEdge == firstSegment.endFaceEdge
        ));

        // Store the amount of segments in the current component
        segmentsPerComponent[component] = componentSegmentIndex;
    }

    // Generate triangle fans for each component
    for (int component = 0; component < 4; component++) {
        if (segmentsPerComponent[component] < 3) continue;
        
        // Solve QEF for component center
        vec3 center = vec3(0.0);
        vec3 centerNormal = vec3(0.0);
        int validCrossings = 0;
        
        // Collect crossings for this component
        for (int i = 0; i < segmentsPerComponent[component]; i++) {
            Segment seg = segmentsByComponent[component][i];
            ivec4 faceEdges = faceToCubeEdgeTable[seg.face];
            int edgeIndex = faceEdges[seg.startFaceEdge];
            
            Crossing crossing = edgeCrossings[edgeIndex];
            center += crossing.position;
            centerNormal += crossing.normal;
            validCrossings++;
        }
        
        // Simple average (replace with proper QEF solver)
        center /= float(validCrossings);
        centerNormal = normalize(centerNormal);
        
        // Get base vertex index
        uint baseVertex = atomicAdd(vertex_count, uint(segmentsPerComponent[component] + 1));
        uint baseIndex = atomicAdd(index_count, uint(segmentsPerComponent[component] * 3));
        
        // Store center vertex
        vertices[baseVertex] = vec4(center, 1.0);
        normals[baseVertex] = vec4(centerNormal, 0.0);
        
        // Store perimeter vertices
        for (int i = 0; i < segmentsPerComponent[component]; i++) {
            Segment seg = segmentsByComponent[component][i];
            ivec4 faceEdges = faceToCubeEdgeTable[seg.face];
            int edgeIndex = faceEdges[seg.startFaceEdge];
            
            Crossing crossing = edgeCrossings[edgeIndex];
            vertices[baseVertex + 1 + i] = vec4(crossing.position, 1.0);
            normals[baseVertex + 1 + i] = vec4(crossing.normal, 0.0);
            
            // Create triangle indices
            if (i > 0) {
                indices[baseIndex + (i-1)*3] = baseVertex;
                indices[baseIndex + (i-1)*3 + 1] = baseVertex + i;
                indices[baseIndex + (i-1)*3 + 2] = baseVertex + i + 1;
            }
        }
        
        // Close the fan
        indices[baseIndex + (segmentsPerComponent[component]-1)*3] = baseVertex;
        indices[baseIndex + (segmentsPerComponent[component]-1)*3 + 1] = baseVertex + segmentsPerComponent[component];
        indices[baseIndex + (segmentsPerComponent[component]-1)*3 + 2] = baseVertex + 1;
    }
}