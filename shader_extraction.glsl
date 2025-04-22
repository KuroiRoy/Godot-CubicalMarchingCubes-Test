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

layout(set = 0, binding = 0, std430) readonly buffer DensityBuffer {
    vec4 densityData[];
};

// Output mesh data
layout(set = 0, binding = 1, std430) buffer VertexBuffer {
    vec4 vertices[];
};

layout(set = 0, binding = 2, std430) buffer IndexBuffer {
    uint indices[];
};

layout(set = 0, binding = 3, std430) buffer CountBuffer {
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
    mat3 AtA;       // A^T * A matrix (3x3)
    vec3 Atb;       // A^T * b vector
    float btb;      // b^T * b scalar
    int numPoints;  // Number of points accumulated
    vec4 massPoint; // Homogeneous mass point (xyz = position, w = count)
};

void qef_initialize(out QEFData qef) {
    qef.AtA = mat3(0.0);
    qef.Atb = vec3(0.0);
    qef.btb = 0.0;
    qef.numPoints = 0;
    qef.massPoint = vec4(0.0);
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
    
    // Accumulate mass point
    qef.massPoint.xyz += p;
    qef.massPoint.w += 1.0;
    qef.numPoints++;
}

vec3 qef_solve_direct(QEFData qef, vec3 massPoint) {
    if (qef.numPoints < 3) return massPoint;
    
    mat3 A = qef.AtA;
    vec3 b = qef.Atb;
    
    // Add small regularization to avoid singularity
    const float REGULARIZATION = 1e-6;
    A[0][0] += REGULARIZATION;
    A[1][1] += REGULARIZATION;
    A[2][2] += REGULARIZATION;
    
    // Calculate determinant (we need to check that it's invertible)
    float det = determinant(A);
    if (abs(det) < 1e-12) {
        return massPoint; // Fallback to mass point if matrix is singular
    }
    
    // Use the inverse of A to solve for x (the best fitting point)
    mat3 A_inv = inverse(A);
    vec3 x = A_inv * b;
    
    // Clamp to cell bounds
    vec3 cellMin = vec3(gl_GlobalInvocationID.xyz);
    vec3 cellMax = cellMin + vec3(1.0);
    return clamp(x, cellMin, cellMax);
}

vec3 qef_solve_iterative(QEFData qef, vec3 massPoint, float svdTolerance, int sweeps) {
    if (qef.numPoints == 0) return massPoint;
    
    mat3 A = qef.AtA;
    vec3 b = qef.Atb;
    
    // Add small regularization to avoid singularity
    const float REGULARIZATION = 1e-6;
    A[0][0] += REGULARIZATION;
    A[1][1] += REGULARIZATION;
    A[2][2] += REGULARIZATION;
    
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
        
        // Update solution using Jacobi iteration
        x += vec3(
            r.x / max(A[0][0], 0.001),
            r.y / max(A[1][1], 0.001),
            r.z / max(A[2][2], 0.001)
        );
    }
    
    // Clamp to cell bounds
    vec3 cellMin = vec3(gl_GlobalInvocationID.xyz);
    vec3 cellMax = cellMin + vec3(1.0);
    return clamp(x, cellMin, cellMax);
}

// Simplified SVD solver for 3x3 symmetric matrices
vec3 qef_solve_svd(QEFData qef, float svdTolerance) {
    if (qef.numPoints < 3) {
        return qef.massPoint.xyz / max(qef.massPoint.w, 1.0);
    }
    
    // Compute mass point (average of positions)
    vec3 massPoint = qef.massPoint.xyz / qef.massPoint.w;
    
    // Regularize the matrix to avoid singularity
    const float REGULARIZATION = 1e-6;
    mat3 A = qef.AtA;
    A[0][0] += REGULARIZATION;
    A[1][1] += REGULARIZATION;
    A[2][2] += REGULARIZATION;
    
    // Compute eigenvalues (simplified approach)
    vec3 eigenvalues;
    {
        // Simple power iteration to estimate dominant eigenvalue
        vec3 v = vec3(1.0, 1.0, 1.0);
        for (int i = 0; i < 4; i++) {
            v = normalize(A * v);
        }
        eigenvalues.x = dot(v, A * v);
        
        // Deflate matrix for next eigenvalue
        mat3 B = A - eigenvalues.x * outerProduct(v, v);
        v = vec3(1.0, -1.0, 1.0);
        for (int i = 0; i < 4; i++) {
            v = normalize(B * v);
        }
        eigenvalues.y = dot(v, B * v);
        
        // Last eigenvalue from trace
        eigenvalues.z = A[0][0] + A[1][1] + A[2][2] - eigenvalues.x - eigenvalues.y;
    }
    
    // Pseudo-inverse threshold
    const float PINV_THRESHOLD = svdTolerance * max(max(eigenvalues.x, eigenvalues.y), eigenvalues.z);
    
    // Compute V*inv(S)*V^T (pseudo-inverse)
    mat3 invA = mat3(0.0);
    if (eigenvalues.x > PINV_THRESHOLD) {
        vec3 v1 = normalize(A * vec3(1,0,0)); // Simplified eigenvector
        invA += (1.0 / eigenvalues.x) * outerProduct(v1, v1);
    }
    if (eigenvalues.y > PINV_THRESHOLD) {
        vec3 v2 = normalize(A * vec3(0,1,0));
        invA += (1.0 / eigenvalues.y) * outerProduct(v2, v2);
    }
    if (eigenvalues.z > PINV_THRESHOLD) {
        vec3 v3 = normalize(A * vec3(0,0,1));
        invA += (1.0 / eigenvalues.z) * outerProduct(v3, v3);
    }
    
    // Solve x = inv(A) * (Atb - AtA*massPoint) + massPoint
    vec3 x = invA * (qef.Atb - A * massPoint) + massPoint;
    
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
    vec4 cornerData[8];
    for (int i = 0; i < 8; i++) {
        ivec3 corner = cellPos + ivec3((i & 1), ((i >> 1) & 1), ((i >> 2) & 1));
        int index = corner.x + corner.y * CHUNK_SIZE + corner.z * CHUNK_SIZE * CHUNK_SIZE;
        cornerData[i] = densityData[index];
    }

    // Find edge crossings
    Crossing edgeCrossings[12];
    for (int edge = 0; edge < 12; edge++) {
        ivec2 corners = cubeEdgeCornerTable[edge];
        vec4 data1 = cornerData[corners.x];
        vec4 data2 = cornerData[corners.y];
        if (data1.x * data2.x > 0.0) continue;

        ivec3 corner1 = cellPos + ivec3((corners.x & 1), ((corners.x >> 1) & 1), ((corners.x >> 2) & 1));
        ivec3 corner2 = cellPos + ivec3((corners.y & 1), ((corners.y >> 1) & 1), ((corners.y >> 2) & 1));
        
        vec3 position = approximateZeroCrossingPosition(corner1, corner2, data1.x, data2.x);
        // vec3 normal = normalize(data1.yzw * abs(data1.x) + data2.yzw * abs(data2.x));
        vec3 normal = normalize(mix(data1.yzw, data2.yzw, abs(data1.x) / (abs(data1.x) + abs(data2.x))));
        
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
        
        if (cornerData[faceCorners.x].x <= 0.0) index |= 1;
        if (cornerData[faceCorners.y].x <= 0.0) index |= 2;
        if (cornerData[faceCorners.z].x <= 0.0) index |= 4;
        if (cornerData[faceCorners.w].x <= 0.0) index |= 8;
        
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
        
        // // Initialize QEF
        // QEFData qef;
        // qef_initialize(qef);
        // vec3 massPoint = vec3(0.0);
        
        // // Accumulate QEF data
        // for (int i = 0; i < segmentsPerComponent[component]; i++) {
        //     Segment seg = segmentsByComponent[component][i];
        //     ivec4 faceEdges = faceToCubeEdgeTable[seg.face];
        //     int edgeIndex = faceEdges[seg.startFaceEdge];
            
        //     Crossing crossing = edgeCrossings[edgeIndex];
        //     qef_add(qef, crossing.position, crossing.normal);
        //     massPoint += crossing.position;
        // }
        
        // // Calculate mass point (average)
        // massPoint /= float(segmentsPerComponent[component]);
        
        // // Solve QEF using direct matrix inversion (more accurate but more expensive)
        // const float QEF_ERROR = 1e-6;
        // vec3 center = qef_solve_direct(qef, massPoint);
        
        // // Alternative: Use iterative solver (faster but less accurate)
        // // const int QEF_SWEEPS = 4;
        // // vec3 center = qef_solve_iterative(qef, massPoint, QEF_ERROR, QEF_SWEEPS);
        
        // // Fallback to mass point if solution is poor
        // float error = length(qef.AtA * center - qef.Atb);
        // if (error > 0.1) {
        //     center = massPoint;
        // }


        // // Initialize QEF
        // QEFData qef;
        // qef_initialize(qef);
        
        // // Accumulate QEF data
        // for (int i = 0; i < segmentsPerComponent[component]; i++) {
        //     Segment seg = segmentsByComponent[component][i];
        //     ivec4 faceEdges = faceToCubeEdgeTable[seg.face];
        //     int edgeIndex = faceEdges[seg.startFaceEdge];
            
        //     Crossing crossing = edgeCrossings[edgeIndex];
        //     qef_add(qef, crossing.position, crossing.normal);
        // }
        
        // // Solve QEF
        // const float QEF_ERROR = 1e-6;
        // vec3 center = qef_solve_svd(qef, QEF_ERROR);
        
        // // Clamp to cell bounds
        // vec3 cellMin = vec3(gl_GlobalInvocationID.xyz);
        // vec3 cellMax = cellMin + vec3(1.0);
        // center = clamp(center, cellMin, cellMax);


        // vec3 averageNormal = vec3(0.0);
        // vec3 weightedCenter = vec3(0.0);
        // float totalWeight = 0.0;

        // for (int i = 0; i < segmentsPerComponent[component]; i++) {
        //     Segment seg = segmentsByComponent[component][i];
        //     ivec4 faceEdges = faceToCubeEdgeTable[seg.face];
        //     int edgeIndex = faceEdges[seg.startFaceEdge];
        //     Crossing crossing = edgeCrossings[edgeIndex];

        //     averageNormal += crossing.normal;
        // }

        // averageNormal = normalize(averageNormal);

        // for (int i = 0; i < segmentsPerComponent[component]; i++) {
        //     Segment seg = segmentsByComponent[component][i];
        //     ivec4 faceEdges = faceToCubeEdgeTable[seg.face];
        //     int edgeIndex = faceEdges[seg.startFaceEdge];
        //     Crossing crossing = edgeCrossings[edgeIndex];

        //     float weight = max(0.01, dot(crossing.normal, averageNormal)); // avoid zero weight
        //     weightedCenter += crossing.position * weight;
        //     totalWeight += weight;
        // }

        // vec3 center = weightedCenter / totalWeight;


        vec3 center = vec3(0.0);
        for (int i = 0; i < segmentsPerComponent[component]; i++) {
            Segment segment = segmentsByComponent[component][i];
            ivec4 faceEdges = faceToCubeEdgeTable[segment.face];
            int edgeIndex = faceEdges[segment.startFaceEdge];
            center += edgeCrossings[edgeIndex].position;
        }
        center /= float(segmentsPerComponent[component]);


        uint iterations = 5;
        float stepSize = 0.1;
        for (int k = 0; k < iterations; k++) {
            vec3 gradient = vec3(0.0);
            
            // Compute gradient (sum of plane corrections)
            for (int i = 0; i < segmentsPerComponent[component]; i++) {
                Segment segment = segmentsByComponent[component][i];
                ivec4 faceEdges = faceToCubeEdgeTable[segment.face];
                int edgeIndex = faceEdges[segment.startFaceEdge];
                Crossing crossing = edgeCrossings[edgeIndex];
                
                // Signed distance from center to the plane (position, normal)
                float distance = dot(crossing.normal, center - crossing.position);
                
                // Correction vector: push center along the normal to minimize distance
                gradient += crossing.normal * distance;
            }
            
            // Nudge center in the opposite direction of the gradient
            center -= stepSize * gradient;
        }

                
        // Get base vertex index
        uint baseVertex = atomicAdd(vertex_count, uint(segmentsPerComponent[component] + 1));
        uint baseIndex = atomicAdd(index_count, uint(segmentsPerComponent[component] * 3));
        
        // Store center vertex
        vertices[baseVertex] = vec4(center, 1.0);
        
        // Store perimeter vertices
        for (int i = 0; i < segmentsPerComponent[component]; i++) {
            Segment seg = segmentsByComponent[component][i];
            ivec4 faceEdges = faceToCubeEdgeTable[seg.face];
            int edgeIndex = faceEdges[seg.startFaceEdge];
            
            Crossing crossing = edgeCrossings[edgeIndex];
            vertices[baseVertex + 1 + i] = vec4(crossing.position, 1.0);
            
            // Create triangle indices
            if (i > 0) {
                indices[baseIndex + (i - 1) * 3] = baseVertex;
                indices[baseIndex + (i - 1) * 3 + 1] = baseVertex + i;
                indices[baseIndex + (i - 1) * 3 + 2] = baseVertex + i + 1;
            }
        }
        
        // Close the fan
        indices[baseIndex + (segmentsPerComponent[component] - 1) * 3] = baseVertex;
        indices[baseIndex + (segmentsPerComponent[component] - 1) * 3 + 1] = baseVertex + segmentsPerComponent[component];
        indices[baseIndex + (segmentsPerComponent[component] - 1) * 3 + 2] = baseVertex + 1;
    }
}