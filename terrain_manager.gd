extends Node3D

const CHUNK_SIZE = 16
const MAX_VERTICES = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * 12

@export var material: Material
@export var compute_shader: Shader

var rd: RenderingDevice
var shader_rid: RID
var pipeline: RID
var uniform_set: RID  # Cache the uniform set

# Buffers
var density_buffer: RID
var vertex_buffer: RID
var index_buffer: RID
var count_buffer: RID

# Mesh tracking
var mesh_instance: MeshInstance3D
var dirty = true
var densityRotation = Quaternion.IDENTITY

# Performance tracking
var frame_count = 0
var last_time = 0.0
var fps = 0.0

func _ready():
	# Initialize rendering device
	rd = RenderingServer.create_local_rendering_device()
	
	# Load compute shader
	var shader_file = load("res://shader_extraction.glsl")
	shader_rid = rd.shader_create_from_spirv(shader_file.get_spirv())
	pipeline = rd.compute_pipeline_create(shader_rid)
	
	# Setup buffers
	setup_buffers()
	create_uniform_set()
	
	# Generate initial density field
	generate_densities()

func _process(delta):
	# Rotate density field
	densityRotation = densityRotation * Quaternion.from_euler(Vector3(0, delta * 0.5, 0))
	
	# Update density field
	generate_densities()
	
	# Update mesh if dirty
	if dirty:
		update_mesh()
		dirty = false
	
	# Calculate FPS
	frame_count += 1
	var current_time = Time.get_ticks_msec() / 1000.0
	if current_time - last_time >= 1.0:
		fps = frame_count / (current_time - last_time)
		print("FPS: ", fps)
		frame_count = 0
		last_time = current_time

func setup_buffers():
	# Density field buffer (float32)
	var density_bytes = PackedByteArray()
	density_bytes.resize(CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * 16) # 16 bytes per vec4 (4 floats × 4 bytes)
	density_buffer = rd.storage_buffer_create(density_bytes.size(), density_bytes)
	
	# Vertex buffer (vec4 = 4 floats)
	var vertex_bytes = PackedByteArray()
	vertex_bytes.resize(MAX_VERTICES * 16) # 16 bytes per vec4 (4 floats × 4 bytes)
	vertex_buffer = rd.storage_buffer_create(vertex_bytes.size(), vertex_bytes)
	
	# Index buffer (uint32)
	var index_bytes = PackedByteArray()
	index_bytes.resize(MAX_VERTICES * 15 * 4) # 4 bytes per uint32
	index_buffer = rd.storage_buffer_create(index_bytes.size(), index_bytes)
	
	# Count buffer (2× uint32)
	var count_bytes = PackedByteArray()
	count_bytes.resize(8) # 8 bytes for 2 uint32s
	count_buffer = rd.storage_buffer_create(count_bytes.size(), count_bytes)

func create_uniform_set():
	# Create uniforms for all bindings (once, not every frame)
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
		
	# Binding 2: Index buffer
	var uniform2 = RDUniform.new()
	uniform2.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform2.binding = 2
	uniform2.add_id(index_buffer)
	uniforms.append(uniform2)
	
	# Binding 3: Count buffer
	var uniform3 = RDUniform.new()
	uniform3.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform3.binding = 3
	uniform3.add_id(count_buffer)
	uniforms.append(uniform3)
	
	# Create uniform set (cached)
	uniform_set = rd.uniform_set_create(uniforms, shader_rid, 0)

func density_function(pos: Vector3) -> float:
	var c = CHUNK_SIZE / 2.0
	var center = Vector3(c, c - 0.2 * CHUNK_SIZE, c)
	var local_pos = densityRotation * (pos - center)

	# Parameters for pyramid shape
	var h = CHUNK_SIZE * 0.6   # height of the pyramid
	var b = CHUNK_SIZE * 0.45   # base width of the pyramid

	# SDF for square-based pyramid
	var q = Vector3(abs(local_pos.x), local_pos.y, abs(local_pos.z))
	var m1 = q.x + q.z
	var m2 = h - q.y
	var sdf = max((m1 * (h / b)) - m2, -local_pos.y)

	return sdf

func density_normal(pos: Vector3) -> Vector3:
	var eps = 0.5
	var dx = density_function(pos + Vector3(eps, 0, 0)) - density_function(pos - Vector3(eps, 0, 0))
	var dy = density_function(pos + Vector3(0, eps, 0)) - density_function(pos - Vector3(0, eps, 0))
	var dz = density_function(pos + Vector3(0, 0, eps)) - density_function(pos - Vector3(0, 0, eps))
	return Vector3(dx, dy, dz).normalized()

func generate_densities():
	# Pre-calculate some values
	var chunk_size_sq = CHUNK_SIZE * CHUNK_SIZE
	var densities = PackedVector4Array()
	densities.resize(CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE)
	
	for z in CHUNK_SIZE:
		for y in CHUNK_SIZE:
			for x in CHUNK_SIZE:
				var idx = x + y * CHUNK_SIZE + z * chunk_size_sq
				# Simple sphere density function
				var pos = Vector3(x, y, z)
				var normal = density_normal(pos);
				densities[idx] = Vector4(density_function(pos), normal.x, normal.y, normal.z)
	
	# Convert to PackedByteArray before uploading
	var byte_data = densities.to_byte_array()
	rd.buffer_update(density_buffer, 0, byte_data.size(), byte_data)
	
	# Mark as dirty to trigger mesh update
	dirty = true

func update_mesh():
	var reset_counts = PackedInt32Array()
	reset_counts.resize(2)
	rd.buffer_update(count_buffer, 0, 8, reset_counts.to_byte_array())

	# Dispatch compute shader using pre-created uniform set
	var compute_list = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline)
	rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
	
	# Dispatch enough workgroups to cover the entire chunk
	var x_groups = CHUNK_SIZE - 1
	var y_groups = CHUNK_SIZE - 1
	var z_groups = CHUNK_SIZE - 1
	
	rd.compute_list_dispatch(compute_list, x_groups, y_groups, z_groups)
	rd.compute_list_end()
	
	rd.submit()
	rd.sync()
	
	# Read back counts
	var counts = rd.buffer_get_data(count_buffer)
	var vertex_count = counts.decode_u32(0)
	var index_count = counts.decode_u32(4)

	print('vertex count: ', vertex_count, ' index count: ', index_count)
	
	if vertex_count == 0:
		if mesh_instance:
			remove_child(mesh_instance)
			mesh_instance.queue_free()
		return
	
	# Read back mesh data
	var vertices_bytes = rd.buffer_get_data(vertex_buffer, 0, vertex_count * 16)
	var indices_bytes = rd.buffer_get_data(index_buffer, 0, index_count * 4)
	
	# Create new mesh
	var arr_mesh = ArrayMesh.new()
	var arrays = []
	arrays.resize(ArrayMesh.ARRAY_MAX)
	
	# Convert bytes to float arrays
	var vertices_floats = vertices_bytes.to_float32_array()
	
	# Convert to Vector3 arrays
	var vertex_array = PackedVector3Array()
	var normal_array = PackedVector3Array()
	
	for i in vertex_count:
		var base = i * 4  # 4 floats per vertex (vec4)
		var vertex = Vector3(
			vertices_floats[base + 0],
			vertices_floats[base + 1],
			vertices_floats[base + 2]
		)
		vertex_array.append(vertex)
		normal_array.append(density_normal(vertex))
	
	# Setup mesh arrays
	arrays[ArrayMesh.ARRAY_VERTEX] = vertex_array
	arrays[ArrayMesh.ARRAY_NORMAL] = normal_array
	arrays[ArrayMesh.ARRAY_INDEX] = indices_bytes.to_int32_array()
	
	# Create mesh
	arr_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	arr_mesh.surface_set_material(0, material)
	
	# Update or create mesh instance
	if mesh_instance:
		mesh_instance.mesh = arr_mesh
	else:
		mesh_instance = MeshInstance3D.new()
		mesh_instance.mesh = arr_mesh
		add_child(mesh_instance)
		mesh_instance.global_position = Vector3.ONE * CHUNK_SIZE / -2.0

func _exit_tree():
	# Clean up RIDs
	if rd:
		rd.free_rid(shader_rid)
		rd.free_rid(pipeline)
		rd.free_rid(density_buffer)
		rd.free_rid(vertex_buffer)
		rd.free_rid(index_buffer)
		rd.free_rid(count_buffer)
		if uniform_set:
			rd.free_rid(uniform_set)
