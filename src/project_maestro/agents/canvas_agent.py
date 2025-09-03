"""Canvas Agent - specialized in generating game art assets using AI image generation."""

import asyncio
import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageOps, ImageFilter
import cv2
import numpy as np

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field

from ..core.agent_framework import BaseAgent, AgentType, AgentTask
from ..core.message_queue import EventType, publish_event
from ..core.config import settings


class ArtRequest(BaseModel):
    """Request for art generation."""
    project_id: str = Field(description="Project identifier")
    asset_type: str = Field(description="Type of asset (character, background, ui, item)")
    style: str = Field(description="Art style description")
    description: str = Field(description="Detailed description of the asset")
    dimensions: Tuple[int, int] = Field(default=(512, 512), description="Image dimensions")
    variations: int = Field(default=1, description="Number of variations to generate")


class GeneratedAsset(BaseModel):
    """Generated art asset."""
    filename: str = Field(description="Asset filename")
    asset_type: str = Field(description="Type of asset")
    dimensions: Tuple[int, int] = Field(description="Image dimensions")
    file_size: int = Field(description="File size in bytes")
    description: str = Field(description="Asset description")
    generation_params: Dict[str, Any] = Field(description="Generation parameters used")


class StableDiffusionAPI:
    """Interface for Stable Diffusion API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.stable_diffusion_api_key
        self.base_url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0"
        
    async def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        cfg_scale: float = 7.0,
        steps: int = 30,
        seed: Optional[int] = None
    ) -> bytes:
        """Generate an image using Stable Diffusion API."""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "text_prompts": [
                {"text": prompt, "weight": 1.0}
            ],
            "cfg_scale": cfg_scale,
            "height": height,
            "width": width,
            "steps": steps,
            "samples": 1
        }
        
        if negative_prompt:
            data["text_prompts"].append({"text": negative_prompt, "weight": -1.0})
            
        if seed:
            data["seed"] = seed
            
        response = requests.post(
            f"{self.base_url}/text-to-image",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            # Decode base64 image
            image_data = base64.b64decode(result["artifacts"][0]["base64"])
            return image_data
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")


class ImageProcessor:
    """Utility class for image processing and optimization."""
    
    @staticmethod
    def create_sprite_sheet(images: List[Image.Image], cols: int = 4) -> Image.Image:
        """Create a sprite sheet from multiple images."""
        if not images:
            raise ValueError("No images provided")
            
        # Calculate dimensions
        img_width, img_height = images[0].size
        rows = (len(images) + cols - 1) // cols
        
        sheet_width = cols * img_width
        sheet_height = rows * img_height
        
        # Create sprite sheet
        sprite_sheet = Image.new('RGBA', (sheet_width, sheet_height), (0, 0, 0, 0))
        
        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            x = col * img_width
            y = row * img_height
            sprite_sheet.paste(img, (x, y))
            
        return sprite_sheet
        
    @staticmethod
    def remove_background(image: Image.Image) -> Image.Image:
        """Remove background using simple color thresholding."""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Define range for background color (assuming white/light background)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Invert mask to get foreground
        mask_inv = cv2.bitwise_not(mask)
        
        # Apply mask to create transparent background
        result = img_array.copy()
        result = cv2.cvtColor(result, cv2.COLOR_RGB2RGBA)
        result[:, :, 3] = mask_inv
        
        return Image.fromarray(result)
        
    @staticmethod
    def optimize_for_mobile(image: Image.Image) -> Image.Image:
        """Optimize image for mobile platforms."""
        # Ensure power-of-2 dimensions for better GPU performance
        width, height = image.size
        
        # Find next power of 2
        def next_power_of_2(n):
            return 2 ** (n - 1).bit_length()
            
        new_width = min(next_power_of_2(width), 1024)  # Max 1024 for mobile
        new_height = min(next_power_of_2(height), 1024)
        
        # Resize with high quality
        optimized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return optimized
        
    @staticmethod
    def create_animation_frames(base_image: Image.Image, frame_count: int = 4) -> List[Image.Image]:
        """Create simple animation frames by applying transformations."""
        frames = []
        
        for i in range(frame_count):
            frame = base_image.copy()
            
            # Apply slight transformations for animation
            if i == 1:
                # Slightly scale up
                frame = frame.resize(
                    (int(frame.width * 1.05), int(frame.height * 1.05)),
                    Image.Resampling.LANCZOS
                )
            elif i == 2:
                # Slight rotation
                frame = frame.rotate(2, expand=False, fillcolor=(0, 0, 0, 0))
            elif i == 3:
                # Slight brightness adjustment
                enhancer = ImageOps.autocontrast(frame)
                frame = enhancer
                
            frames.append(frame)
            
        return frames


class CanvasTools:
    """Tools available to the Canvas Agent."""
    
    @staticmethod
    def create_generate_character_tool(agent: "CanvasAgent") -> BaseTool:
        """Tool for generating character sprites."""
        
        class CharacterInput(BaseModel):
            name: str = Field(description="Character name")
            description: str = Field(description="Character description")
            art_style: str = Field(description="Art style")
            animations: List[str] = Field(default=["idle", "walk"], description="Required animations")
            
        async def generate_character(
            name: str,
            description: str,
            art_style: str,
            animations: List[str]
        ) -> str:
            try:
                asset = await agent._generate_character_sprites(
                    name, description, art_style, animations
                )
                return f"Generated {name} character with {len(animations)} animations"
            except Exception as e:
                return f"Error generating character: {str(e)}"
                
        return StructuredTool.from_function(
            func=generate_character,
            name="generate_character",
            description="Generate character sprites with animations",
            args_schema=CharacterInput
        )
        
    @staticmethod
    def create_generate_background_tool(agent: "CanvasAgent") -> BaseTool:
        """Tool for generating background assets."""
        
        class BackgroundInput(BaseModel):
            name: str = Field(description="Background name")
            description: str = Field(description="Background description")
            art_style: str = Field(description="Art style")
            tileable: bool = Field(default=False, description="Should be tileable")
            
        async def generate_background(
            name: str,
            description: str,
            art_style: str,
            tileable: bool
        ) -> str:
            try:
                asset = await agent._generate_background(
                    name, description, art_style, tileable
                )
                return f"Generated {name} background ({'tileable' if tileable else 'single'})"
            except Exception as e:
                return f"Error generating background: {str(e)}"
                
        return StructuredTool.from_function(
            func=generate_background,
            name="generate_background",
            description="Generate background environment assets",
            args_schema=BackgroundInput
        )
        
    @staticmethod
    def create_generate_ui_elements_tool(agent: "CanvasAgent") -> BaseTool:
        """Tool for generating UI elements."""
        
        class UIElementsInput(BaseModel):
            elements: List[str] = Field(description="List of UI elements to generate")
            art_style: str = Field(description="UI art style")
            color_scheme: str = Field(description="Color scheme")
            
        async def generate_ui_elements(
            elements: List[str],
            art_style: str,
            color_scheme: str
        ) -> str:
            try:
                assets = await agent._generate_ui_elements(
                    elements, art_style, color_scheme
                )
                return f"Generated {len(assets)} UI elements"
            except Exception as e:
                return f"Error generating UI elements: {str(e)}"
                
        return StructuredTool.from_function(
            func=generate_ui_elements,
            name="generate_ui_elements",
            description="Generate UI interface elements",
            args_schema=UIElementsInput
        )


class CanvasAgent(BaseAgent):
    """Specialist agent for generating game art assets."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="canvas_agent",
            agent_type=AgentType.CANVAS,
            **kwargs
        )
        
        self.sd_api = StableDiffusionAPI()
        self.processor = ImageProcessor()
        self.generated_assets: Dict[str, GeneratedAsset] = {}
        
        # Create tools
        self.tools = [
            CanvasTools.create_generate_character_tool(self),
            CanvasTools.create_generate_background_tool(self),
            CanvasTools.create_generate_ui_elements_tool(self),
        ]
        
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the LangChain agent executor for art generation."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            callbacks=[self.callback_handler]
        )
        
    def get_system_prompt(self) -> str:
        """Get the system prompt for the Canvas agent."""
        return """
        You are the Canvas Agent, a specialist in generating high-quality game art assets using AI.
        
        Your expertise includes:
        - Character design and sprite creation
        - Environment and background art
        - UI/UX design elements
        - Game icons and items
        - Animation frame generation
        - Art style consistency
        
        Art generation capabilities:
        - 2D sprite art in various styles (pixel art, cartoon, realistic, hand-drawn)
        - Character design with multiple poses and animations
        - Environment backgrounds (parallax-ready layers)
        - UI elements (buttons, panels, icons)
        - Tileable textures and patterns
        - Mobile-optimized assets
        
        Quality standards:
        - Consistent art style within projects
        - Mobile-optimized dimensions (power-of-2 textures)
        - Transparent backgrounds for sprites
        - Clear, readable designs
        - Appropriate color palettes
        - Performance-conscious file sizes
        
        Technical considerations:
        - Generate assets in common game formats (PNG with transparency)
        - Create sprite sheets for character animations
        - Ensure assets work well at different resolutions
        - Consider memory usage and loading times
        - Follow platform-specific guidelines (iOS, Android)
        
        Always maintain artistic coherence across all assets within a single project.
        """
        
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute an art generation task."""
        
        action = task.action
        params = task.parameters
        
        if action == "generate_character_sprites":
            return await self._handle_character_generation(params)
        elif action == "generate_environment_assets":
            return await self._handle_environment_generation(params)
        elif action == "generate_ui_assets":
            return await self._handle_ui_generation(params)
        else:
            # Use agent executor for complex art generation
            result = await self.agent_executor.ainvoke({
                "input": f"Generate art assets for: {action} with specifications: {params}"
            })
            return {"result": result["output"]}
            
    async def _handle_character_generation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle character sprite generation."""
        
        character = params.get("character", {})
        art_style = params.get("art_style", "pixel art")
        project_id = params.get("project_id", "")
        
        name = character.get("name", "character")
        description = character.get("description", "game character")
        animations = character.get("animations", ["idle", "walk"])
        
        self.log_agent_action("generate_character_sprites", "started",
                            character=name, project_id=project_id)
        
        try:
            asset = await self._generate_character_sprites(
                name, description, art_style, animations
            )
            
            # Store asset
            asset_key = f"{project_id}_{name}_sprites"
            self.generated_assets[asset_key] = asset
            
            # Publish asset generated event
            await publish_event(
                EventType.ASSET_GENERATED,
                "canvas_agent",
                {
                    "asset_type": "character_sprites",
                    "filename": asset.filename,
                    "character": name,
                    "project_id": project_id,
                    "animations": animations
                }
            )
            
            return {
                "filename": asset.filename,
                "asset_type": asset.asset_type,
                "dimensions": asset.dimensions,
                "animations_count": len(animations)
            }
            
        except Exception as e:
            self.log_agent_error("generate_character_sprites", e,
                               character=name, project_id=project_id)
            raise
            
    async def _generate_character_sprites(
        self,
        name: str,
        description: str,
        art_style: str,
        animations: List[str]
    ) -> GeneratedAsset:
        """Generate character sprite sheets."""
        
        # Create detailed prompt for character
        base_prompt = f"""
        {art_style} style game character sprite, {description}, 
        clean transparent background, centered character,
        game asset, high quality, consistent lighting,
        suitable for mobile games, clear details
        """
        
        negative_prompt = """
        blurry, low quality, text, watermark, signature,
        background elements, multiple characters, cropped
        """
        
        all_frames = []
        
        # Generate frames for each animation
        for animation in animations:
            animation_prompt = f"{base_prompt}, {animation} pose"
            
            if animation == "idle":
                animation_prompt += ", standing still, neutral pose"
            elif animation == "walk":
                animation_prompt += ", walking cycle, side view"
            elif animation == "jump":
                animation_prompt += ", jumping pose, dynamic action"
            elif animation == "attack":
                animation_prompt += ", attack pose, action stance"
                
            try:
                # Generate base image
                image_data = await self.sd_api.generate_image(
                    prompt=animation_prompt,
                    negative_prompt=negative_prompt,
                    width=512,
                    height=512,
                    cfg_scale=7.0,
                    steps=25
                )
                
                # Process image
                base_image = Image.open(io.BytesIO(image_data))
                base_image = self.processor.remove_background(base_image)
                base_image = self.processor.optimize_for_mobile(base_image)
                
                # Create animation frames
                frames = self.processor.create_animation_frames(base_image, 4)
                all_frames.extend(frames)
                
            except Exception as e:
                self.logger.error(f"Failed to generate {animation} animation", error=str(e))
                # Create placeholder frame
                placeholder = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
                all_frames.extend([placeholder] * 4)
                
        # Create sprite sheet
        sprite_sheet = self.processor.create_sprite_sheet(all_frames, cols=4)
        
        # Save sprite sheet
        filename = f"{name}_sprites.png"
        output_path = settings.data_dir / "assets" / "characters" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sprite_sheet.save(output_path, "PNG")
        
        return GeneratedAsset(
            filename=filename,
            asset_type="character_sprites",
            dimensions=sprite_sheet.size,
            file_size=output_path.stat().st_size,
            description=f"Character sprites for {name} with {len(animations)} animations",
            generation_params={
                "art_style": art_style,
                "animations": animations,
                "base_prompt": base_prompt
            }
        )
        
    async def _handle_environment_generation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle environment asset generation."""
        
        environment = params.get("environment", {})
        art_style = params.get("art_style", "pixel art")
        project_id = params.get("project_id", "")
        
        name = environment.get("name", "environment")
        description = environment.get("description", "game environment")
        
        asset = await self._generate_background(name, description, art_style, False)
        
        # Store and publish event
        asset_key = f"{project_id}_{name}_bg"
        self.generated_assets[asset_key] = asset
        
        await publish_event(
            EventType.ASSET_GENERATED,
            "canvas_agent",
            {
                "asset_type": "background",
                "filename": asset.filename,
                "environment": name,
                "project_id": project_id
            }
        )
        
        return {
            "filename": asset.filename,
            "asset_type": asset.asset_type,
            "dimensions": asset.dimensions
        }
        
    async def _generate_background(
        self,
        name: str,
        description: str,
        art_style: str,
        tileable: bool
    ) -> GeneratedAsset:
        """Generate background environment."""
        
        prompt = f"""
        {art_style} style game background, {description},
        detailed environment, game asset, high quality,
        suitable for mobile games, atmospheric lighting,
        no characters, landscape view
        """
        
        if tileable:
            prompt += ", seamless tileable pattern, repeating texture"
            
        negative_prompt = """
        characters, people, animals, text, watermark,
        UI elements, buttons, low quality, blurry
        """
        
        try:
            # Generate background
            image_data = await self.sd_api.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=1024,
                height=512,  # Landscape format
                cfg_scale=6.0,
                steps=30
            )
            
            # Process image
            background = Image.open(io.BytesIO(image_data))
            
            if tileable:
                # Make tileable by blending edges
                background = self._make_tileable(background)
                
            background = self.processor.optimize_for_mobile(background)
            
            # Save background
            suffix = "_tileable" if tileable else "_bg"
            filename = f"{name}{suffix}.png"
            output_path = settings.data_dir / "assets" / "backgrounds" / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            background.save(output_path, "PNG")
            
            return GeneratedAsset(
                filename=filename,
                asset_type="background",
                dimensions=background.size,
                file_size=output_path.stat().st_size,
                description=f"Background environment: {description}",
                generation_params={
                    "art_style": art_style,
                    "tileable": tileable,
                    "prompt": prompt
                }
            )
            
        except Exception as e:
            self.log_agent_error("generate_background", e)
            raise
            
    def _make_tileable(self, image: Image.Image) -> Image.Image:
        """Make an image tileable by blending edges."""
        # Simple edge blending approach
        width, height = image.size
        blend_width = width // 10
        
        # Create masks for blending
        left_mask = np.linspace(0, 1, blend_width)
        right_mask = np.linspace(1, 0, blend_width)
        
        img_array = np.array(image)
        
        # Blend left and right edges
        for i in range(blend_width):
            img_array[:, i] = (
                img_array[:, i] * left_mask[i] +
                img_array[:, width - blend_width + i] * (1 - left_mask[i])
            )
            img_array[:, width - blend_width + i] = (
                img_array[:, width - blend_width + i] * right_mask[i] +
                img_array[:, i] * (1 - right_mask[i])
            )
            
        return Image.fromarray(img_array.astype(np.uint8))
        
    async def _handle_ui_generation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle UI asset generation."""
        
        elements = params.get("elements", [])
        art_style = params.get("art_style", "clean modern")
        color_scheme = params.get("color_scheme", "blue")
        project_id = params.get("project_id", "")
        
        assets = await self._generate_ui_elements(elements, art_style, color_scheme)
        
        return {
            "generated_assets": len(assets),
            "elements": elements
        }
        
    async def _generate_ui_elements(
        self,
        elements: List[str],
        art_style: str,
        color_scheme: str
    ) -> List[GeneratedAsset]:
        """Generate UI elements."""
        
        assets = []
        
        for element in elements:
            prompt = f"""
            {art_style} {color_scheme} {element} UI element,
            game interface, clean design, mobile-friendly,
            transparent background, high quality icon,
            simple and clear
            """
            
            try:
                image_data = await self.sd_api.generate_image(
                    prompt=prompt,
                    width=256,
                    height=256,
                    cfg_scale=5.0,
                    steps=20
                )
                
                ui_image = Image.open(io.BytesIO(image_data))
                ui_image = self.processor.remove_background(ui_image)
                
                # Save UI element
                filename = f"{element}_ui.png"
                output_path = settings.data_dir / "assets" / "ui" / filename
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                ui_image.save(output_path, "PNG")
                
                asset = GeneratedAsset(
                    filename=filename,
                    asset_type="ui_element",
                    dimensions=ui_image.size,
                    file_size=output_path.stat().st_size,
                    description=f"UI element: {element}",
                    generation_params={"element": element, "style": art_style}
                )
                
                assets.append(asset)
                
            except Exception as e:
                self.logger.error(f"Failed to generate {element} UI", error=str(e))
                
        return assets