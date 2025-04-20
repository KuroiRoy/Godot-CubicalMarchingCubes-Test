extends Node3D

const CHUNK_SIZE = 32
const MAX_VERTICES = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * 12

@export var material: Material
@export var compute_shader: Shader

var rd: RenderingDevice
var shader_rid: RID
var pipeline: RID

# Buffers
var hermite_buffer: RID
var vertex_buffer: RID
var normal_buffer: RID
var index_buffer: RID
var count_buffer: RID

func _ready():
	# Initialize rendering device
	rd = RenderingServer.create_local_rendering_device()
	
	# Load compute shader
	var shader_file = load("res://shader_extraction.glsl")
	shader_rid = rd.shader_create_from_spirv(shader_file.get_spirv())
	pipeline = rd.compute_pipeline_create(shader_rid)
	
	# Setup buffers
	setup_buffers()
	
	# Generate test density field
	generate_test_density()
	
	# Run compute shader
	extract_surface()
	
	# Create mesh from results
	create_mesh()

func setup_buffers():
	# Density field buffer (float32)
	var hermite_bytes = PackedByteArray()
	# hermite_bytes.resize(CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * 7 * 4)  # 7 floats * 4 bytes per each
	hermite_bytes.resize(CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * 3 * 12) # 12 bytes per element (std430 uses largest member, in this case vec3)
	hermite_buffer = rd.storage_buffer_create(hermite_bytes.size(), hermite_bytes)
	
	# Vertex buffer (vec4 = 4 floats)
	var vertex_bytes = PackedByteArray()
	vertex_bytes.resize(MAX_VERTICES * 16)  # 16 bytes per vec4 (4 floats × 4 bytes)
	vertex_buffer = rd.storage_buffer_create(vertex_bytes.size(), vertex_bytes)
	
	# Normal buffer (vec4 = 4 floats)
	var normal_bytes = PackedByteArray()
	normal_bytes.resize(MAX_VERTICES * 16)  # Same as vertex buffer
	normal_buffer = rd.storage_buffer_create(normal_bytes.size(), normal_bytes)
	
	# Index buffer (uint32)
	var index_bytes = PackedByteArray()
	index_bytes.resize(MAX_VERTICES * 15 * 4)  # 4 bytes per uint32
	index_buffer = rd.storage_buffer_create(index_bytes.size(), index_bytes)
	
	# Count buffer (2× uint32)
	var count_bytes = PackedByteArray()
	count_bytes.resize(8)  # 8 bytes for 2 uint32s
	count_buffer = rd.storage_buffer_create(count_bytes.size(), count_bytes)

func generate_test_density():
	# Create arrays for all components of HermiteData
	var positions = PackedFloat32Array()
	var normals = PackedFloat32Array()
	var densities = PackedFloat32Array()
	
	# Each array will contain CHUNK_SIZE^3 elements (but positions and normals have 3 components each)
	positions.resize(CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * 3)
	normals.resize(CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * 3)
	densities.resize(CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE)
	
	var center = Vector3(CHUNK_SIZE / 2.0, CHUNK_SIZE / 2.0, CHUNK_SIZE / 2.0)
	
	for z in CHUNK_SIZE:
		for y in CHUNK_SIZE:
			for x in CHUNK_SIZE:
				var pos = Vector3(x, y, z)
				var idx = x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE
				var vec_idx = idx * 3
				
				# Position (3 floats)
				positions[vec_idx] = pos.x
				positions[vec_idx + 1] = pos.y
				positions[vec_idx + 2] = pos.z
				
				# Density (1 float)
				var dist = pos.distance_to(center)
				densities[idx] = dist - (CHUNK_SIZE / 3.0)
				
				# Normal (3 floats) - gradient of the density field
				var normal = (pos - center).normalized()
				normals[vec_idx] = normal.x
				normals[vec_idx + 1] = normal.y
				normals[vec_idx + 2] = normal.z
	
	# Combine all data into a single byte array matching the HermiteData struct
	var byte_data = PackedByteArray()
	# byte_data.resize(CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * 7 * 4)  # 7 floats per element
	byte_data.resize(CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * 3 * 12) # 12 bytes per element (std430 uses largest member, in this case vec3)
	
	# Manually pack the data in the correct order (position, normal, density)
	for i in range(CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE):
		var base_offset = i * 3 * 12 # 12 bytes per element
		
		# Position (3 floats)
		for j in range(3):
			var bytes = positions.to_byte_array().slice((i * 3 + j) * 4, (i * 3 + j + 1) * 4)
			for k in range(4):
				byte_data[base_offset + j * 4 + k] = bytes[k]
		
		# Normal (3 floats)
		for j in range(3):
			var bytes = normals.to_byte_array().slice((i * 3 + j) * 4, (i * 3 + j + 1) * 4)
			for k in range(4):
				byte_data[base_offset + 12 + j * 4 + k] = bytes[k]  # 12 = 3 floats * 4 bytes
		
		# Density (1 float)
		var density_bytes = densities.to_byte_array().slice(i * 4, (i + 1) * 4)
		for j in range(4):
			byte_data[base_offset + 24 + j] = density_bytes[j]  # 24 = 6 floats * 4 bytes
	
	rd.buffer_update(hermite_buffer, 0, byte_data.size(), byte_data)

func extract_surface():
	# Create uniforms for all bindings
	var uniforms = []
	
	# Binding 0: Density buffer
	var uniform0 = RDUniform.new()
	uniform0.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform0.binding = 0
	uniform0.add_id(hermite_buffer)
	uniforms.append(uniform0)
	
	# Binding 1: Vertex buffer
	var uniform1 = RDUniform.new()
	uniform1.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform1.binding = 1
	uniform1.add_id(vertex_buffer)
	uniforms.append(uniform1)
	
	# Binding 2: Normal buffer
	var uniform2 = RDUniform.new()
	uniform2.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform2.binding = 2
	uniform2.add_id(normal_buffer)
	uniforms.append(uniform2)
	
	# Binding 3: Index buffer
	var uniform3 = RDUniform.new()
	uniform3.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform3.binding = 3
	uniform3.add_id(index_buffer)
	uniforms.append(uniform3)
	
	# Binding 4: Count buffer
	var uniform4 = RDUniform.new()
	uniform4.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform4.binding = 4
	uniform4.add_id(count_buffer)
	uniforms.append(uniform4)
	
	# Create uniform set with all uniforms
	var uniform_set = rd.uniform_set_create(uniforms, shader_rid, 0)
	
	# Dispatch compute shader
	var compute_list = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline)
	rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
	
	# Dispatch enough workgroups to cover the entire chunk
	var x_groups = CHUNK_SIZE - 1
	var y_groups = CHUNK_SIZE - 1
	var z_groups = CHUNK_SIZE - 1
	
	rd.compute_list_dispatch(compute_list, x_groups, y_groups, z_groups)
	rd.compute_list_end()
	
	# Force synchronization (for demo purposes - in production you'd want async)
	rd.submit()
	rd.sync()

func create_mesh():
	# Read back counts
	var counts = rd.buffer_get_data(count_buffer)
	var vertex_count = counts.decode_u32(0)
	var index_count = counts.decode_u32(4)
	
	print('vertex count: ', vertex_count, ' index count: ', index_count)
	
	if vertex_count == 0:
		return
	
	# Read back mesh data
	var vertices_bytes = rd.buffer_get_data(vertex_buffer, 0, vertex_count * 16)
	var normals_bytes = rd.buffer_get_data(normal_buffer, 0, vertex_count * 16)
	var indices_bytes = rd.buffer_get_data(index_buffer, 0, index_count * 4)
	
	# Create Godot mesh
	var arr_mesh = ArrayMesh.new()
	var arrays = []
	arrays.resize(ArrayMesh.ARRAY_MAX)
		
	# Convert bytes to float arrays
	var vertices_floats = vertices_bytes.to_float32_array()
	var normals_floats = normals_bytes.to_float32_array()
	
	# Convert to Vector3 arrays
	var vertex_array = PackedVector3Array()
	var normal_array = PackedVector3Array()
	
	for i in vertex_count:
		var base = i * 4  # 4 floats per vertex (vec4)
		vertex_array.append(Vector3(
			vertices_floats[base + 0],
			vertices_floats[base + 1],
			vertices_floats[base + 2]
		))
		normal_array.append(Vector3(
			normals_floats[base + 0],
			normals_floats[base + 1],
			normals_floats[base + 2]
		))
	
	# Convert indices
	var index_array = indices_bytes.to_int32_array()
	
	# Setup mesh arrays
	arrays[ArrayMesh.ARRAY_VERTEX] = vertex_array
	arrays[ArrayMesh.ARRAY_NORMAL] = normal_array
	arrays[ArrayMesh.ARRAY_INDEX] = index_array
	
	# Create mesh
	arr_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	arr_mesh.surface_set_material(0, material)
	
	# Create mesh instance
	var mesh_instance = MeshInstance3D.new()
	mesh_instance.mesh = arr_mesh
	add_child(mesh_instance)

func _exit_tree():
	# Clean up RIDs
	if rd:
		rd.free_rid(shader_rid)
		rd.free_rid(pipeline)
		rd.free_rid(hermite_buffer)
		rd.free_rid(vertex_buffer)
		rd.free_rid(normal_buffer)
		rd.free_rid(index_buffer)
		rd.free_rid(count_buffer)
