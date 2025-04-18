extends Node3D

const CHUNK_SIZE = 32
const MAX_VERTICES = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * 5 # Conservative estimate

@export var material: Material
@export var compute_shader: Shader

var rd: RenderingDevice
var shader_rid: RID
var pipeline: RID

# Buffers
var density_buffer: RID
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
	var density_bytes = PackedByteArray()
	density_bytes.resize(CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * 4)  # 4 bytes per float
	density_buffer = rd.storage_buffer_create(density_bytes.size(), density_bytes)
	
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
	index_bytes.resize(MAX_VERTICES * 4)  # 4 bytes per uint32
	index_buffer = rd.storage_buffer_create(index_bytes.size(), index_bytes)
	
	# Count buffer (2× uint32)
	var count_bytes = PackedByteArray()
	count_bytes.resize(8)  # 8 bytes for 2 uint32s
	count_buffer = rd.storage_buffer_create(count_bytes.size(), count_bytes)

func generate_test_density():
	var densities = PackedFloat32Array()
	densities.resize(CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE)
	
	for z in CHUNK_SIZE:
		for y in CHUNK_SIZE:
			for x in CHUNK_SIZE:
				var idx = x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE
				# Simple sphere density function
				var center = Vector3(CHUNK_SIZE/2, CHUNK_SIZE/2, CHUNK_SIZE/2)
				var pos = Vector3(x, y, z)
				densities[idx] = pos.distance_to(center) - (CHUNK_SIZE/3)
	
	# Convert to PackedByteArray before uploading
	var byte_data = densities.to_byte_array()
	rd.buffer_update(density_buffer, 0, byte_data.size(), byte_data)

func extract_surface():
	# Create uniforms for all bindings
	var uniforms = []
	
	# Binding 0: Density buffer
	var uniform0 = RDUniform.new()
	uniform0.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform0.binding = 0
	uniform0.add_id(density_buffer)
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
	var x_groups = (CHUNK_SIZE + 7) / 8
	var y_groups = (CHUNK_SIZE + 7) / 8
	var z_groups = (CHUNK_SIZE + 7) / 8
	
	rd.compute_list_dispatch(compute_list, x_groups, y_groups, z_groups)
	rd.compute_list_end()
	
	# Force synchronization (for demo purposes - in production you'd want async)
	rd.submit()
	rd.sync()

func create_mesh():
	# Read back counts
	var counts = rd.buffer_get_data(count_buffer)
	var vertex_count = counts.decode_s32(0)
	var index_count = counts.decode_s32(4)
	
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
		rd.free_rid(density_buffer)
		rd.free_rid(vertex_buffer)
		rd.free_rid(normal_buffer)
		rd.free_rid(index_buffer)
		rd.free_rid(count_buffer)
