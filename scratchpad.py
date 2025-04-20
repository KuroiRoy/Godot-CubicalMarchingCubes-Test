from glm import ivec2, ivec4

faceNameTable = [
    "Left",
    "Right",
    "Up",
    "Down",
    "Forward",
    "Back"
]

cubeEdgeCornerTable = [
    ivec2(2, 0), ivec2(0, 1), ivec2(1, 3), ivec2(3, 2),
    ivec2(6, 4), ivec2(4, 5), ivec2(5, 7), ivec2(7, 6),
    ivec2(0, 4), ivec2(1, 5), ivec2(3, 7), ivec2(2, 6)
]

faceToCubeEdgeTable = [
    ivec4(4, 0, 11, 8),   # Left
    ivec4(2, 6, 10, 9),   # Right
    ivec4(11, 10, 7, 3),  # Up
    ivec4(8, 9, 1, 5),    # Down
    ivec4(6, 4, 7, 5),    # Forward
    ivec4(0, 2, 3, 1)     # Back
]

faceToCubeCornerTable = [
    ivec4(4, 0, 6, 2),    # Left
    ivec4(1, 5, 3, 7),    # Right
    ivec4(2, 3, 6, 7),    # Up
    ivec4(4, 5, 0, 1),    # Down
    ivec4(5, 4, 7, 6),    # Forward
    ivec4(0, 1, 2, 3)     # Back
]

nextFaceTable = [
    ivec2(4, 1), ivec2(5, 0), ivec2(2, 0), ivec2(3, 0), # Left    -> Forward, Back,    Up,      Down
    ivec2(5, 1), ivec2(4, 0), ivec2(2, 1), ivec2(3, 1), # Right   -> Back,    Forward, Up,      Down
    ivec2(0, 2), ivec2(1, 2), ivec2(4, 2), ivec2(5, 2), # Up      -> Left,    Right,   Forward, Back
    ivec2(0, 3), ivec2(1, 3), ivec2(5, 3), ivec2(4, 3), # Down    -> Left,    Right,   Back,    Forward
    ivec2(1, 1), ivec2(0, 0), ivec2(2, 2), ivec2(3, 3), # Forward -> Right,   Left,    Up,      Down
    ivec2(0, 1), ivec2(1, 0), ivec2(2, 3), ivec2(3, 2)  # Back    -> Left,    Right,   Up,      Down 
]


for (face, name) in enumerate(faceNameTable):
    for edge in [0, 1, 2, 3]:
        nextFace = nextFaceTable[face * 4 + edge]
        print(str(faceNameTable[face]).ljust(7), str(edge).rjust(2), '  ', str(faceNameTable[nextFace.x]).ljust(7), nextFace.y)