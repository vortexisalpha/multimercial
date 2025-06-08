"""
Rendering Engine Module

Implements Physically Based Rendering (PBR) and advanced compositing for
realistic advertisement placement in video scenes.
"""

import logging
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import json

import numpy as np
import cv2
from PIL import Image
import moderngl as mgl
import pyrr

from ..utils.logging_utils import get_logger
from ..utils.geometry_utils import compute_homography, project_3d_to_2d
from ..utils.rendering_utils import (
    create_shader_program,
    load_texture,
    setup_lighting,
    compute_pbr_materials
)

logger = get_logger(__name__)


class RenderingMode(Enum):
    """Rendering modes for advertisement placement."""
    BASIC = "basic"
    PBR = "pbr"
    ADVANCED = "advanced"


class BlendingMode(Enum):
    """Blending modes for compositing."""
    ALPHA = "alpha"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    NORMAL = "normal"


@dataclass
class AdAsset:
    """Advertisement asset for rendering."""
    image: np.ndarray
    alpha_mask: Optional[np.ndarray] = None
    position: Tuple[float, float, float] = (0, 0, 0)
    rotation: Tuple[float, float, float] = (0, 0, 0)
    scale: Tuple[float, float] = (1.0, 1.0)
    opacity: float = 1.0
    blending_mode: BlendingMode = BlendingMode.ALPHA


@dataclass
class RenderResult:
    """Result of rendering operation."""
    rendered_image: np.ndarray
    depth_buffer: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None


class RenderingConfig:
    """Configuration for rendering engine."""
    
    def __init__(
        self,
        rendering_mode: RenderingMode = RenderingMode.PBR,
        output_resolution: Tuple[int, int] = (1920, 1080),
        enable_shadows: bool = True,
        enable_reflections: bool = True,
        enable_ambient_occlusion: bool = True,
        shadow_quality: str = "high",  # low, medium, high
        lighting_model: str = "pbr",  # pbr, phong, lambert
        enable_tone_mapping: bool = True,
        enable_color_correction: bool = True,
        msaa_samples: int = 4,
        **kwargs
    ):
        self.rendering_mode = rendering_mode
        self.output_resolution = output_resolution
        self.enable_shadows = enable_shadows
        self.enable_reflections = enable_reflections
        self.enable_ambient_occlusion = enable_ambient_occlusion
        self.shadow_quality = shadow_quality
        self.lighting_model = lighting_model
        self.enable_tone_mapping = enable_tone_mapping
        self.enable_color_correction = enable_color_correction
        self.msaa_samples = msaa_samples
        self.kwargs = kwargs


class RenderingEngine:
    """
    Advanced rendering engine for photorealistic advertisement placement.
    
    This class provides comprehensive 3D rendering capabilities including
    PBR materials, advanced lighting, shadows, and compositing.
    
    Attributes:
        config: Rendering configuration
        context: ModernGL rendering context
        shaders: Compiled shader programs
        framebuffers: Rendering framebuffers
        is_initialized: Initialization status
    """

    def __init__(self, config: RenderingConfig):
        """
        Initialize the rendering engine with specified configuration.
        
        Args:
            config: RenderingConfig with rendering parameters
        """
        self.config = config
        self.context: Optional[mgl.Context] = None
        self.shaders: Dict[str, mgl.Program] = {}
        self.framebuffers: Dict[str, mgl.Framebuffer] = {}
        self.textures: Dict[str, mgl.Texture] = {}
        self.is_initialized = False
        
        logger.info("Initializing RenderingEngine")
        self._initialize_graphics_context()

    def _initialize_graphics_context(self) -> None:
        """Initialize ModernGL graphics context."""
        try:
            # Create offscreen context for headless rendering
            self.context = mgl.create_context(standalone=True)
            
            # Setup framebuffers
            self._setup_framebuffers()
            
            # Load and compile shaders
            self._load_shaders()
            
            # Setup default textures
            self._setup_default_textures()
            
            self.is_initialized = True
            logger.info("Graphics context initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize graphics context: {str(e)}")
            raise RuntimeError(f"Graphics initialization failed: {str(e)}")

    def _setup_framebuffers(self) -> None:
        """Setup rendering framebuffers."""
        width, height = self.config.output_resolution
        
        # Main color buffer
        color_texture = self.context.texture((width, height), 4)  # RGBA
        depth_texture = self.context.depth_texture((width, height))
        
        # Main framebuffer
        self.framebuffers['main'] = self.context.framebuffer(
            [color_texture], depth_texture
        )
        
        # Shadow map framebuffer
        if self.config.enable_shadows:
            shadow_size = 2048 if self.config.shadow_quality == "high" else 1024
            shadow_depth = self.context.depth_texture((shadow_size, shadow_size))
            self.framebuffers['shadow'] = self.context.framebuffer([], shadow_depth)
        
        # Post-processing framebuffer
        post_color = self.context.texture((width, height), 4)
        self.framebuffers['post'] = self.context.framebuffer([post_color])

    def _load_shaders(self) -> None:
        """Load and compile shader programs."""
        # Vertex shader for basic geometry
        vertex_shader = """
        #version 330 core
        in vec3 position;
        in vec2 texcoord;
        in vec3 normal;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform mat4 light_space_matrix;
        
        out vec2 uv;
        out vec3 world_pos;
        out vec3 world_normal;
        out vec4 light_space_pos;
        
        void main() {
            vec4 world_position = model * vec4(position, 1.0);
            world_pos = world_position.xyz;
            world_normal = mat3(transpose(inverse(model))) * normal;
            light_space_pos = light_space_matrix * world_position;
            
            uv = texcoord;
            gl_Position = projection * view * world_position;
        }
        """
        
        # PBR fragment shader
        pbr_fragment = """
        #version 330 core
        in vec2 uv;
        in vec3 world_pos;
        in vec3 world_normal;
        in vec4 light_space_pos;
        
        uniform sampler2D albedo_texture;
        uniform sampler2D normal_texture;
        uniform sampler2D metallic_texture;
        uniform sampler2D roughness_texture;
        uniform sampler2D ao_texture;
        uniform sampler2D shadow_map;
        
        uniform vec3 camera_pos;
        uniform vec3 light_pos;
        uniform vec3 light_color;
        uniform float light_intensity;
        
        out vec4 fragment_color;
        
        vec3 calculate_pbr_lighting() {
            // Simplified PBR calculation
            vec3 albedo = texture(albedo_texture, uv).rgb;
            float metallic = texture(metallic_texture, uv).r;
            float roughness = texture(roughness_texture, uv).r;
            float ao = texture(ao_texture, uv).r;
            
            vec3 N = normalize(world_normal);
            vec3 V = normalize(camera_pos - world_pos);
            vec3 L = normalize(light_pos - world_pos);
            vec3 H = normalize(V + L);
            
            // Calculate lighting
            float NdotL = max(dot(N, L), 0.0);
            float NdotV = max(dot(N, V), 0.0);
            float NdotH = max(dot(N, H), 0.0);
            float VdotH = max(dot(V, H), 0.0);
            
            // Fresnel
            vec3 F0 = mix(vec3(0.04), albedo, metallic);
            vec3 F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
            
            // Distribution
            float alpha = roughness * roughness;
            float alpha2 = alpha * alpha;
            float denom = NdotH * NdotH * (alpha2 - 1.0) + 1.0;
            float D = alpha2 / (3.14159265 * denom * denom);
            
            // Geometry
            float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
            float G1L = NdotL / (NdotL * (1.0 - k) + k);
            float G1V = NdotV / (NdotV * (1.0 - k) + k);
            float G = G1L * G1V;
            
            // BRDF
            vec3 numerator = D * G * F;
            float denominator = 4.0 * NdotV * NdotL + 0.001;
            vec3 specular = numerator / denominator;
            
            vec3 kS = F;
            vec3 kD = vec3(1.0) - kS;
            kD *= 1.0 - metallic;
            
            vec3 diffuse = kD * albedo / 3.14159265;
            
            return (diffuse + specular) * light_color * light_intensity * NdotL * ao;
        }
        
        float calculate_shadow() {
            vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
            proj_coords = proj_coords * 0.5 + 0.5;
            
            if (proj_coords.z > 1.0) return 0.0;
            
            float closest_depth = texture(shadow_map, proj_coords.xy).r;
            float current_depth = proj_coords.z;
            
            float bias = 0.005;
            return current_depth - bias > closest_depth ? 1.0 : 0.0;
        }
        
        void main() {
            vec3 color = calculate_pbr_lighting();
            
            if (textureSize(shadow_map, 0).x > 1) {
                float shadow = calculate_shadow();
                color *= (1.0 - shadow * 0.5);
            }
            
            // Tone mapping
            color = color / (color + vec3(1.0));
            color = pow(color, vec3(1.0/2.2));  // Gamma correction
            
            fragment_color = vec4(color, 1.0);
        }
        """
        
        # Compile shaders
        try:
            self.shaders['pbr'] = self.context.program(
                vertex_shader=vertex_shader,
                fragment_shader=pbr_fragment
            )
            
            # Simple quad shader for post-processing
            quad_vertex = """
            #version 330 core
            in vec2 position;
            out vec2 uv;
            
            void main() {
                uv = position * 0.5 + 0.5;
                gl_Position = vec4(position, 0.0, 1.0);
            }
            """
            
            quad_fragment = """
            #version 330 core
            in vec2 uv;
            uniform sampler2D scene_texture;
            out vec4 fragment_color;
            
            void main() {
                fragment_color = texture(scene_texture, uv);
            }
            """
            
            self.shaders['quad'] = self.context.program(
                vertex_shader=quad_vertex,
                fragment_shader=quad_fragment
            )
            
        except Exception as e:
            logger.error(f"Shader compilation failed: {str(e)}")
            raise

    def _setup_default_textures(self) -> None:
        """Setup default textures for rendering."""
        # White texture
        white_data = np.ones((1, 1, 3), dtype=np.uint8) * 255
        self.textures['white'] = self.context.texture((1, 1), 3, white_data.tobytes())
        
        # Normal map texture
        normal_data = np.array([[[128, 128, 255]]], dtype=np.uint8)
        self.textures['default_normal'] = self.context.texture((1, 1), 3, normal_data.tobytes())

    def composite_ads(
        self,
        background_image: np.ndarray,
        depth_map: np.ndarray,
        objects: List,
        planes: List,
        camera_params: Any,
        ad_assets: List[AdAsset]
    ) -> RenderResult:
        """
        Composite advertisements into the scene with realistic rendering.
        
        Args:
            background_image: Original video frame
            depth_map: Scene depth information
            objects: Detected objects in scene
            planes: Detected planes for placement
            camera_params: Camera parameters
            ad_assets: List of advertisement assets to render
            
        Returns:
            RenderResult with composited image and metadata
        """
        if not self.is_initialized:
            raise RuntimeError("Rendering engine not initialized")

        try:
            logger.info(f"Compositing {len(ad_assets)} advertisement assets")
            
            # Setup scene
            self._setup_scene(background_image, depth_map, camera_params)
            
            # Render advertisements
            rendered_ads = []
            for ad_asset in ad_assets:
                placement_plane = self._find_best_placement_plane(ad_asset, planes)
                if placement_plane:
                    rendered_ad = self._render_ad_on_plane(ad_asset, placement_plane, camera_params)
                    rendered_ads.append(rendered_ad)
            
            # Composite final result
            final_image = self._composite_final_image(background_image, rendered_ads)
            
            # Generate metadata
            metadata = {
                "num_ads_rendered": len(rendered_ads),
                "rendering_mode": self.config.rendering_mode.value,
                "resolution": self.config.output_resolution,
                "camera_params": camera_params.get_model_info() if hasattr(camera_params, 'get_model_info') else None
            }
            
            return RenderResult(
                rendered_image=final_image,
                depth_buffer=depth_map,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Advertisement compositing failed: {str(e)}")
            raise RuntimeError(f"Compositing error: {str(e)}")

    def _setup_scene(self, image: np.ndarray, depth_map: np.ndarray, camera_params: Any) -> None:
        """Setup 3D scene for rendering."""
        # Clear framebuffer
        self.framebuffers['main'].use()
        self.context.clear(0.0, 0.0, 0.0, 1.0)
        self.context.enable(mgl.DEPTH_TEST)
        
        # Setup camera matrices
        if hasattr(camera_params, 'intrinsics'):
            intrinsics = camera_params.intrinsics
            view_matrix = self._compute_view_matrix(camera_params.extrinsics)
            proj_matrix = self._compute_projection_matrix(intrinsics, image.shape[:2])
        else:
            # Default matrices
            view_matrix = pyrr.matrix44.create_look_at(
                eye=[0, 0, 5], target=[0, 0, 0], up=[0, 1, 0]
            )
            proj_matrix = pyrr.matrix44.create_perspective_projection_matrix(
                60.0, image.shape[1] / image.shape[0], 0.1, 100.0
            )

        # Upload matrices to shaders
        for shader in self.shaders.values():
            if 'view' in shader:
                shader['view'].write(view_matrix.astype(np.float32).tobytes())
            if 'projection' in shader:
                shader['projection'].write(proj_matrix.astype(np.float32).tobytes())

    def _find_best_placement_plane(self, ad_asset: AdAsset, planes: List) -> Optional[Any]:
        """Find the best plane for placing an advertisement."""
        if not planes:
            return None
        
        # Score planes based on suitability for advertisement placement
        scored_planes = []
        for plane in planes:
            score = self._compute_placement_score(plane, ad_asset)
            scored_planes.append((plane, score))
        
        # Sort by score and return best plane
        scored_planes.sort(key=lambda x: x[1], reverse=True)
        
        return scored_planes[0][0] if scored_planes[0][1] > 0.5 else None

    def _compute_placement_score(self, plane: Any, ad_asset: AdAsset) -> float:
        """Compute suitability score for placing ad on a plane."""
        score = 0.0
        
        # Prefer larger planes
        area_score = min(getattr(plane, 'area', 0) / 2.0, 1.0)
        
        # Prefer vertical planes (walls) for most ads
        orientation_score = 1.0 if getattr(plane, 'orientation', None) == 'vertical' else 0.7
        
        # Factor in confidence
        confidence_score = getattr(plane, 'confidence', 0.5)
        
        # Check texture quality if available
        texture_score = getattr(plane, 'texture_quality', 0.8)
        
        score = (area_score * 0.3 + orientation_score * 0.3 + 
                confidence_score * 0.2 + texture_score * 0.2)
        
        return min(max(score, 0.0), 1.0)

    def _render_ad_on_plane(self, ad_asset: AdAsset, plane: Any, camera_params: Any) -> np.ndarray:
        """Render advertisement on a specific plane."""
        try:
            # Create geometry for the ad quad
            ad_geometry = self._create_ad_geometry(ad_asset, plane)
            
            # Setup ad texture
            ad_texture = self._create_texture_from_image(ad_asset.image)
            
            # Bind textures and uniforms
            self.shaders['pbr']['albedo_texture'] = 0
            ad_texture.use(0)
            
            # Set material properties
            self._setup_ad_material_properties(ad_asset)
            
            # Render the ad
            ad_geometry.render()
            
            # Read back result
            result = self._read_framebuffer()
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to render ad on plane: {str(e)}")
            return np.zeros((100, 100, 3), dtype=np.uint8)

    def _create_ad_geometry(self, ad_asset: AdAsset, plane: Any) -> Any:
        """Create 3D geometry for advertisement placement."""
        # Extract plane properties
        center = getattr(plane, 'center', np.array([0, 0, 0]))
        normal = getattr(plane, 'normal', np.array([0, 0, 1]))
        
        # Calculate quad size based on ad aspect ratio
        ad_height, ad_width = ad_asset.image.shape[:2]
        aspect_ratio = ad_width / ad_height
        
        # Default ad size in world units
        world_height = 1.0
        world_width = world_height * aspect_ratio
        
        # Create quad vertices
        # This is simplified - in practice you'd need proper plane-to-quad mapping
        vertices = np.array([
            [-world_width/2, -world_height/2, 0, 0, 0],  # position + texcoord
            [ world_width/2, -world_height/2, 0, 1, 0],
            [ world_width/2,  world_height/2, 0, 1, 1],
            [-world_width/2,  world_height/2, 0, 0, 1],
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        
        # Create VAO
        vbo = self.context.buffer(vertices.tobytes())
        ibo = self.context.buffer(indices.tobytes())
        
        vao = self.context.vertex_array(
            self.shaders['pbr'],
            [(vbo, '3f 2f', 'position', 'texcoord')],
            ibo
        )
        
        return vao

    def _create_texture_from_image(self, image: np.ndarray) -> mgl.Texture:
        """Create OpenGL texture from image array."""
        if len(image.shape) == 3:
            height, width, channels = image.shape
        else:
            height, width = image.shape
            channels = 1
        
        # Convert to RGB if needed
        if channels == 3:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Create texture
        texture = self.context.texture((width, height), channels, image_rgb.tobytes())
        texture.filter = (mgl.LINEAR, mgl.LINEAR)
        
        return texture

    def _setup_ad_material_properties(self, ad_asset: AdAsset) -> None:
        """Setup material properties for advertisement rendering."""
        shader = self.shaders['pbr']
        
        # Basic material properties
        if 'metallic_texture' in shader:
            self.textures['white'].use(1)
            shader['metallic_texture'] = 1
        
        if 'roughness_texture' in shader:
            self.textures['white'].use(2)
            shader['roughness_texture'] = 2
        
        if 'normal_texture' in shader:
            self.textures['default_normal'].use(3)
            shader['normal_texture'] = 3

    def _composite_final_image(self, background: np.ndarray, rendered_ads: List[np.ndarray]) -> np.ndarray:
        """Composite rendered advertisements with background."""
        result = background.copy()
        
        for ad_image in rendered_ads:
            # Simple alpha blending - in practice you'd use depth buffer
            # and proper compositing based on the rendering
            if ad_image.size > 0:
                # Resize ad to match background if needed
                if ad_image.shape[:2] != background.shape[:2]:
                    ad_image = cv2.resize(ad_image, (background.shape[1], background.shape[0]))
                
                # Alpha blend
                alpha = 0.8  # Ad opacity
                result = cv2.addWeighted(result, 1 - alpha, ad_image, alpha, 0)
        
        return result

    def _compute_view_matrix(self, extrinsics: Any) -> np.ndarray:
        """Compute view matrix from camera extrinsics."""
        if hasattr(extrinsics, 'transformation_matrix'):
            # Invert camera transform to get view matrix
            transform = extrinsics.transformation_matrix
            view_matrix = np.linalg.inv(transform)
        else:
            # Default view matrix
            view_matrix = pyrr.matrix44.create_look_at(
                eye=[0, 0, 5], target=[0, 0, 0], up=[0, 1, 0]
            )
        
        return view_matrix.astype(np.float32)

    def _compute_projection_matrix(self, intrinsics: Any, image_size: Tuple[int, int]) -> np.ndarray:
        """Compute projection matrix from camera intrinsics."""
        height, width = image_size
        
        if hasattr(intrinsics, 'fx'):
            fx, fy = intrinsics.fx, intrinsics.fy
            cx, cy = intrinsics.cx, intrinsics.cy
            
            # Convert to OpenGL projection matrix
            near, far = 0.1, 100.0
            
            projection = np.array([
                [2*fx/width, 0, (width - 2*cx)/width, 0],
                [0, 2*fy/height, (2*cy - height)/height, 0],
                [0, 0, -(far + near)/(far - near), -2*far*near/(far - near)],
                [0, 0, -1, 0]
            ], dtype=np.float32)
        else:
            # Default perspective projection
            projection = pyrr.matrix44.create_perspective_projection_matrix(
                60.0, width / height, 0.1, 100.0
            )
        
        return projection.astype(np.float32)

    def _read_framebuffer(self) -> np.ndarray:
        """Read rendered result from framebuffer."""
        width, height = self.config.output_resolution
        
        # Read pixels from framebuffer
        data = self.framebuffers['main'].read(components=3)
        
        # Convert to numpy array and flip vertically
        image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        image = np.flipud(image)  # OpenGL to image coordinates
        
        return image

    def render_shadow_map(self, light_pos: np.ndarray, scene_objects: List) -> None:
        """Render shadow map for realistic shadows."""
        if not self.config.enable_shadows or 'shadow' not in self.framebuffers:
            return
        
        # Switch to shadow framebuffer
        self.framebuffers['shadow'].use()
        self.context.clear()
        
        # Setup light view and projection matrices
        light_view = pyrr.matrix44.create_look_at(
            eye=light_pos, target=[0, 0, 0], up=[0, 1, 0]
        )
        light_proj = pyrr.matrix44.create_orthogonal_projection_matrix(
            -10, 10, -10, 10, 1.0, 20.0
        )
        
        # Render scene from light's perspective
        # This would render all scene geometry to create depth map
        pass

    def apply_post_processing(self, image: np.ndarray) -> np.ndarray:
        """Apply post-processing effects."""
        if not self.config.enable_tone_mapping and not self.config.enable_color_correction:
            return image
        
        result = image.copy().astype(np.float32) / 255.0
        
        # Tone mapping
        if self.config.enable_tone_mapping:
            result = result / (result + 1.0)  # Reinhard tone mapping
        
        # Color correction
        if self.config.enable_color_correction:
            # Simple gamma correction
            result = np.power(result, 1.0/2.2)
        
        return (result * 255.0).astype(np.uint8)

    def cleanup(self) -> None:
        """Clean up rendering resources."""
        if self.context:
            # Clean up textures
            for texture in self.textures.values():
                texture.release()
            self.textures.clear()
            
            # Clean up framebuffers
            for fb in self.framebuffers.values():
                fb.release()
            self.framebuffers.clear()
            
            # Clean up shaders
            for shader in self.shaders.values():
                shader.release()
            self.shaders.clear()
            
            # Release context
            self.context.release()
            self.context = None
        
        self.is_initialized = False
        logger.info("RenderingEngine cleanup completed") 