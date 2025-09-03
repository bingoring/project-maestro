"""Builder Agent - specialized in Unity integration and automated game building."""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field

from ..core.agent_framework import BaseAgent, AgentType, AgentTask
from ..core.message_queue import EventType, publish_event
from ..core.storage import get_asset_manager
from ..core.config import settings
from ..core.logging import get_logger


class UnityProjectTemplate:
    """Template for creating Unity projects."""
    
    @staticmethod
    def create_project_structure(project_path: Path) -> None:
        """Create basic Unity project structure."""
        # Create directory structure
        dirs_to_create = [
            "Assets",
            "Assets/Scripts",
            "Assets/Sprites",
            "Assets/Audio",
            "Assets/Audio/BGM",
            "Assets/Audio/SFX",
            "Assets/Prefabs",
            "Assets/Scenes",
            "Assets/Materials",
            "Assets/Animations",
            "Packages",
            "ProjectSettings",
            "UserSettings"
        ]
        
        for dir_name in dirs_to_create:
            (project_path / dir_name).mkdir(parents=True, exist_ok=True)
            
    @staticmethod
    def create_project_settings(project_path: Path, project_name: str) -> None:
        """Create essential Unity project settings."""
        
        # ProjectVersion.txt
        version_file = project_path / "ProjectSettings" / "ProjectVersion.txt"
        version_file.write_text("m_EditorVersion: 2023.2.0f1\nm_EditorVersionWithRevision: 2023.2.0f1\n")
        
        # ProjectSettings.asset (simplified)
        project_settings = {
            "PlayerSettings": {
                "productName": project_name,
                "companyName": "Project Maestro",
                "bundleVersion": "1.0.0",
                "targetDevice": "Handheld"
            }
        }
        
        # EditorBuildSettings.asset
        build_settings = project_path / "ProjectSettings" / "EditorBuildSettings.asset"
        build_settings.write_text("""scenes:
- enabled: 1
  path: Assets/Scenes/MainScene.unity
  guid: 0000000000000000000000000000000001""")
        
    @staticmethod
    def create_main_scene(project_path: Path) -> None:
        """Create a basic main scene."""
        scene_content = """%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!29 &1
OcclusionCullingSettings:
  m_ObjectHideFlags: 0
  serializedVersion: 2
  m_OcclusionBakeSettings:
    smallestOccluder: 5
    smallestHole: 0.25
    backfaceThreshold: 100
--- !u!104 &2
RenderSettings:
  m_ObjectHideFlags: 0
  serializedVersion: 9
  m_Fog: 0
  m_FogColor: {r: 0.5, g: 0.5, b: 0.5, a: 1}
  m_FogMode: 3
  m_FogDensity: 0.01
  m_LinearFogStart: 0
  m_LinearFogEnd: 300
  m_AmbientSkyColor: {r: 0.212, g: 0.227, b: 0.259, a: 1}
  m_AmbientEquatorColor: {r: 0.114, g: 0.125, b: 0.133, a: 1}
  m_AmbientGroundColor: {r: 0.047, g: 0.043, b: 0.035, a: 1}
  m_AmbientIntensity: 1
  m_AmbientMode: 3
  m_SubtractiveShadowColor: {r: 0.42, g: 0.478, b: 0.627, a: 1}
  m_SkyboxMaterial: {fileID: 0}
  m_HaloStrength: 0.5
  m_FlareStrength: 1
  m_FlareFadeSpeed: 3
  m_HaloTexture: {fileID: 0}
  m_SpotCookie: {fileID: 10001, guid: 0000000000000000e000000000000000, type: 0}
  m_DefaultReflectionMode: 0
  m_DefaultReflectionResolution: 128
  m_ReflectionBounces: 1
  m_ReflectionIntensity: 1
  m_CustomReflection: {fileID: 0}
  m_Sun: {fileID: 0}
  m_IndirectSpecularColor: {r: 0, g: 0, b: 0, a: 1}
--- !u!157 &3
LightmapSettings:
  m_ObjectHideFlags: 0
  serializedVersion: 12
  m_GIWorkflowMode: 1
--- !u!196 &4
NavMeshSettings:
  serializedVersion: 2
  m_ObjectHideFlags: 0
  m_BuildSettings:
    serializedVersion: 2
    agentTypeID: 0
    agentRadius: 0.5
    agentHeight: 2
    agentSlope: 45
    agentClimb: 0.4
    ledgeDropHeight: 0
    maxJumpAcrossDistance: 0
    minRegionArea: 2
    manualCellSize: 0
    cellSize: 0.16666667
    manualTileSize: 0
    tileSize: 256
    accuratePlacement: 0
    debug:
      m_Flags: 0
  m_NavMeshData: {fileID: 0}
--- !u!1 &705507993
GameObject:
  m_ObjectHideFlags: 0
  m_CorrespondingSourceObject: {fileID: 0}
  m_PrefabInstance: {fileID: 0}
  m_PrefabAsset: {fileID: 0}
  serializedVersion: 6
  m_Component:
  - component: {fileID: 705507995}
  - component: {fileID: 705507994}
  m_Layer: 0
  m_Name: Directional Light
  m_TagString: Untagged
  m_Icon: {fileID: 0}
  m_NavMeshLayer: 0
  m_StaticEditorFlags: 0
  m_IsActive: 1
--- !u!108 &705507994
Light:
  m_ObjectHideFlags: 0
  m_CorrespondingSourceObject: {fileID: 0}
  m_PrefabInstance: {fileID: 0}
  m_PrefabAsset: {fileID: 0}
  m_GameObject: {fileID: 705507993}
  m_Enabled: 1
  serializedVersion: 10
  m_Type: 1
  m_Shape: 0
  m_Color: {r: 1, g: 0.95686275, b: 0.8392157, a: 1}
  m_Intensity: 1
  m_Range: 10
  m_SpotAngle: 30
  m_InnerSpotAngle: 21.80208
  m_CookieSize: 10
  m_Shadows:
    m_Type: 2
    m_Resolution: -1
    m_CustomResolution: -1
    m_Strength: 1
    m_Bias: 0.05
    m_NormalBias: 0.4
    m_NearPlane: 0.2
    m_CullingMatrixOverride:
      e00: 1
      e01: 0
      e02: 0
      e03: 0
      e10: 0
      e11: 1
      e12: 0
      e13: 0
      e20: 0
      e21: 0
      e22: 1
      e23: 0
      e30: 0
      e31: 0
      e32: 0
      e33: 1
    m_UseCullingMatrixOverride: 0
  m_Cookie: {fileID: 0}
  m_DrawHalo: 0
  m_Flare: {fileID: 0}
  m_RenderMode: 0
  m_CullingMask:
    serializedVersion: 2
    m_Bits: 4294967295
  m_RenderingLayerMask: 1
  m_Lightmapping: 1
  m_LightShadowCasterMode: 0
  m_AreaSize: {x: 1, y: 1}
  m_BounceIntensity: 1
  m_ColorTemperature: 6570
  m_UseColorTemperature: 0
  m_BoundingSphereOverride: {x: 0, y: 0, z: 0, w: 0}
  m_UseBoundingSphereOverride: 0
  m_ShadowRadius: 0
  m_ShadowAngle: 0
--- !u!4 &705507995
Transform:
  m_ObjectHideFlags: 0
  m_CorrespondingSourceObject: {fileID: 0}
  m_PrefabInstance: {fileID: 0}
  m_PrefabAsset: {fileID: 0}
  m_GameObject: {fileID: 705507993}
  m_LocalRotation: {x: 0.40821788, y: -0.23456968, z: 0.10938163, w: 0.8754261}
  m_LocalPosition: {x: 0, y: 3, z: 0}
  m_LocalScale: {x: 1, y: 1, z: 1}
  m_Children: []
  m_Father: {fileID: 0}
  m_RootOrder: 1
  m_LocalEulerAnglesHint: {x: 50, y: -30, z: 0}
--- !u!1 &963194225
GameObject:
  m_ObjectHideFlags: 0
  m_CorrespondingSourceObject: {fileID: 0}
  m_PrefabInstance: {fileID: 0}
  m_PrefabAsset: {fileID: 0}
  serializedVersion: 6
  m_Component:
  - component: {fileID: 963194228}
  - component: {fileID: 963194227}
  - component: {fileID: 963194226}
  m_Layer: 0
  m_Name: Main Camera
  m_TagString: MainCamera
  m_Icon: {fileID: 0}
  m_NavMeshLayer: 0
  m_StaticEditorFlags: 0
  m_IsActive: 1
--- !u!81 &963194226
AudioListener:
  m_ObjectHideFlags: 0
  m_CorrespondingSourceObject: {fileID: 0}
  m_PrefabInstance: {fileID: 0}
  m_PrefabAsset: {fileID: 0}
  m_GameObject: {fileID: 963194225}
  m_Enabled: 1
--- !u!20 &963194227
Camera:
  m_ObjectHideFlags: 0
  m_CorrespondingSourceObject: {fileID: 0}
  m_PrefabInstance: {fileID: 0}
  m_PrefabAsset: {fileID: 0}
  m_GameObject: {fileID: 963194225}
  m_Enabled: 1
  serializedVersion: 2
  m_ClearFlags: 1
  m_BackGroundColor: {r: 0.19215687, g: 0.3019608, b: 0.4745098, a: 0}
  m_projectionMatrixMode: 1
  m_GateFitMode: 2
  m_FOVAxisMode: 0
  m_SensorSize: {x: 36, y: 24}
  m_LensShift: {x: 0, y: 0}
  m_FocalLength: 50
  m_NormalizedViewPortRect:
    serializedVersion: 2
    x: 0
    y: 0
    width: 1
    height: 1
  m_near: 0.3
  m_far: 1000
  m_orthographic: 0
  m_orthographicSize: 5
  m_depth: -1
  m_cullingMask:
    serializedVersion: 2
    m_Bits: 4294967295
  m_renderingPath: -1
  m_targetTexture: {fileID: 0}
  m_targetDisplay: 0
  m_targetEye: 3
  m_HDR: 1
  m_AllowMSAA: 1
  m_AllowDynamicResolution: 0
  m_ForceIntoRT: 0
  m_OcclusionCulling: 1
  m_StereoConvergence: 10
  m_StereoSeparation: 0.022
--- !u!4 &963194228
Transform:
  m_ObjectHideFlags: 0
  m_CorrespondingSourceObject: {fileID: 0}
  m_PrefabInstance: {fileID: 0}
  m_PrefabAsset: {fileID: 0}
  m_GameObject: {fileID: 963194225}
  m_LocalRotation: {x: 0, y: 0, z: 0, w: 1}
  m_LocalPosition: {x: 0, y: 1, z: -10}
  m_LocalScale: {x: 1, y: 1, z: 1}
  m_Children: []
  m_Father: {fileID: 0}
  m_RootOrder: 0
  m_LocalEulerAnglesHint: {x: 0, y: 0, z: 0}
"""
        
        scene_path = project_path / "Assets" / "Scenes" / "MainScene.unity"
        scene_path.write_text(scene_content)


class UnityBuilder:
    """Handles Unity project building and automation."""
    
    def __init__(self, unity_path: str):
        self.unity_path = Path(unity_path)
        self.logger = get_logger("unity_builder")
        
    async def create_project(
        self,
        project_name: str,
        project_path: Path
    ) -> bool:
        """Create a new Unity project."""
        
        try:
            self.logger.info("Creating Unity project", name=project_name, path=str(project_path))
            
            # Create project directory structure
            project_path.mkdir(parents=True, exist_ok=True)
            UnityProjectTemplate.create_project_structure(project_path)
            UnityProjectTemplate.create_project_settings(project_path, project_name)
            UnityProjectTemplate.create_main_scene(project_path)
            
            self.logger.info("Unity project created successfully")
            return True
            
        except Exception as e:
            self.logger.error("Failed to create Unity project", error=str(e))
            return False
            
    async def import_assets(
        self,
        project_path: Path,
        assets: List[Tuple[Path, str]]  # (source_path, destination_relative_path)
    ) -> bool:
        """Import assets into Unity project."""
        
        try:
            for source_path, dest_relative_path in assets:
                dest_path = project_path / "Assets" / dest_relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                if source_path.is_file():
                    shutil.copy2(source_path, dest_path)
                elif source_path.is_dir():
                    shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                    
                self.logger.info(
                    "Asset imported",
                    source=str(source_path),
                    destination=dest_relative_path
                )
                
            return True
            
        except Exception as e:
            self.logger.error("Failed to import assets", error=str(e))
            return False
            
    async def build_project(
        self,
        project_path: Path,
        build_target: str = "Android",
        build_path: Optional[Path] = None
    ) -> Optional[Path]:
        """Build Unity project for target platform."""
        
        if not build_path:
            build_path = settings.unity_build_path / f"build_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        build_path.mkdir(parents=True, exist_ok=True)
        
        try:
            self.logger.info(
                "Starting Unity build",
                project=str(project_path),
                target=build_target,
                output=str(build_path)
            )
            
            # Unity command line arguments
            unity_args = [
                str(self.unity_path),
                "-batchmode",
                "-quit",
                "-projectPath", str(project_path),
                "-buildTarget", build_target,
                "-buildPath", str(build_path / f"game.{self._get_extension_for_target(build_target)}"),
                "-executeMethod", "BuildScript.Build"
            ]
            
            # Create build script if it doesn't exist
            await self._create_build_script(project_path, build_target)
            
            # Execute Unity build
            process = await asyncio.create_subprocess_exec(
                *unity_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=project_path
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info(
                    "Unity build completed successfully",
                    output_path=str(build_path)
                )
                return build_path
            else:
                self.logger.error(
                    "Unity build failed",
                    return_code=process.returncode,
                    stdout=stdout.decode(),
                    stderr=stderr.decode()
                )
                return None
                
        except Exception as e:
            self.logger.error("Unity build error", error=str(e))
            return None
            
    async def _create_build_script(self, project_path: Path, build_target: str):
        """Create Unity build script."""
        
        build_script_content = f"""
using UnityEngine;
using UnityEditor;
using UnityEditor.Build.Reporting;

public class BuildScript
{{
    public static void Build()
    {{
        string[] scenes = {{"Assets/Scenes/MainScene.unity"}};
        string buildPath = System.Environment.GetCommandLineArgs()[System.Array.IndexOf(System.Environment.GetCommandLineArgs(), "-buildPath") + 1];
        
        BuildPlayerOptions buildPlayerOptions = new BuildPlayerOptions();
        buildPlayerOptions.scenes = scenes;
        buildPlayerOptions.locationPathName = buildPath;
        buildPlayerOptions.target = BuildTarget.{build_target};
        buildPlayerOptions.options = BuildOptions.None;
        
        BuildReport report = BuildPipeline.BuildPlayer(buildPlayerOptions);
        BuildSummary summary = report.summary;
        
        if (summary.result == BuildResult.Succeeded)
        {{
            Debug.Log("Build succeeded: " + summary.totalSize + " bytes");
        }}
        
        if (summary.result == BuildResult.Failed)
        {{
            Debug.Log("Build failed");
        }}
        
        EditorApplication.Exit(summary.result == BuildResult.Succeeded ? 0 : 1);
    }}
}}
"""
        
        script_path = project_path / "Assets" / "Editor"
        script_path.mkdir(exist_ok=True)
        
        (script_path / "BuildScript.cs").write_text(build_script_content)
        
    def _get_extension_for_target(self, build_target: str) -> str:
        """Get file extension for build target."""
        extensions = {
            "Android": "apk",
            "iOS": "ipa",
            "StandaloneWindows": "exe",
            "StandaloneWindows64": "exe",
            "StandaloneOSX": "app",
            "WebGL": "html"
        }
        return extensions.get(build_target, "bin")


class BuilderTools:
    """Tools available to the Builder Agent."""
    
    @staticmethod
    def create_build_game_tool(agent: "BuilderAgent") -> BaseTool:
        """Tool for building complete games."""
        
        class BuildGameInput(BaseModel):
            project_id: str = Field(description="Project identifier")
            project_name: str = Field(description="Game project name")
            build_target: str = Field(description="Build target platform (Android, iOS, WebGL)")
            
        async def build_game(
            project_id: str,
            project_name: str,
            build_target: str
        ) -> str:
            try:
                build_result = await agent._build_complete_game(
                    project_id, project_name, build_target
                )
                if build_result:
                    return f"Successfully built {project_name} for {build_target}"
                else:
                    return f"Failed to build {project_name}"
            except Exception as e:
                return f"Error building game: {str(e)}"
                
        return StructuredTool.from_function(
            func=build_game,
            name="build_game",
            description="Build a complete game from generated assets",
            args_schema=BuildGameInput
        )
        
    @staticmethod
    def create_create_project_tool(agent: "BuilderAgent") -> BaseTool:
        """Tool for creating Unity projects."""
        
        class CreateProjectInput(BaseModel):
            project_name: str = Field(description="Unity project name")
            template: str = Field(description="Project template (2d, 3d, mobile)")
            
        async def create_project(
            project_name: str,
            template: str
        ) -> str:
            try:
                success = await agent._create_unity_project(project_name, template)
                return f"{'Successfully created' if success else 'Failed to create'} Unity project: {project_name}"
            except Exception as e:
                return f"Error creating project: {str(e)}"
                
        return StructuredTool.from_function(
            func=create_project,
            name="create_project",
            description="Create a new Unity project",
            args_schema=CreateProjectInput
        )
        
    @staticmethod
    def create_import_assets_tool(agent: "BuilderAgent") -> BaseTool:
        """Tool for importing assets into Unity."""
        
        class ImportAssetsInput(BaseModel):
            project_id: str = Field(description="Project identifier")
            asset_types: List[str] = Field(description="Types of assets to import")
            
        async def import_assets(
            project_id: str,
            asset_types: List[str]
        ) -> str:
            try:
                count = await agent._import_project_assets(project_id, asset_types)
                return f"Successfully imported {count} assets for project {project_id}"
            except Exception as e:
                return f"Error importing assets: {str(e)}"
                
        return StructuredTool.from_function(
            func=import_assets,
            name="import_assets",
            description="Import generated assets into Unity project",
            args_schema=ImportAssetsInput
        )


class BuilderAgent(BaseAgent):
    """Specialist agent for Unity integration and automated game building."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="builder_agent",
            agent_type=AgentType.BUILDER,
            **kwargs
        )
        
        self.unity_builder = UnityBuilder(settings.unity_path)
        self.asset_manager = get_asset_manager()
        self.active_projects: Dict[str, Path] = {}
        
        # Create tools
        self.tools = [
            BuilderTools.create_build_game_tool(self),
            BuilderTools.create_create_project_tool(self),
            BuilderTools.create_import_assets_tool(self),
        ]
        
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the LangChain agent executor for building."""
        
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
        """Get the system prompt for the Builder agent."""
        return """
        You are the Builder Agent, the final specialist responsible for integrating all generated 
        assets into a working Unity game and building the final executable.
        
        Your expertise includes:
        - Unity project setup and configuration
        - Asset integration and organization
        - Build pipeline automation
        - Cross-platform deployment
        - Performance optimization for mobile platforms
        - Quality assurance and testing
        
        Integration capabilities:
        - Automated Unity project creation
        - Script integration and compilation
        - Sprite and audio asset importing
        - Scene construction and prefab creation
        - UI system setup and connection
        - Build configuration for multiple platforms
        
        Build pipeline:
        - Android APK generation
        - iOS IPA creation (with proper provisioning)
        - WebGL builds for web deployment  
        - Standalone builds for desktop testing
        - Automated testing and validation
        
        Quality standards:
        - Error-free compilation
        - Proper asset references and connections
        - Optimized builds for target platforms
        - Clean project structure and organization
        - Performance validation on target devices
        
        Technical considerations:
        - Mobile-specific optimizations
        - Memory management and loading strategies
        - Platform-specific build configurations
        - Asset compression and streaming
        - Error handling and crash prevention
        
        Always ensure the final build is stable, performant, and ready for distribution.
        """
        
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute a build task."""
        
        action = task.action
        params = task.parameters
        
        if action == "build_game":
            return await self._handle_build_game(params)
        elif action == "create_project":
            return await self._handle_create_project(params)
        elif action == "import_assets":
            return await self._handle_import_assets(params)
        else:
            # Use agent executor for complex build tasks
            result = await self.agent_executor.ainvoke({
                "input": f"Build task: {action} with parameters: {params}"
            })
            return {"result": result["output"]}
            
    async def _handle_build_game(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle complete game building."""
        
        project_spec = params.get("project_spec", {})
        project_id = params.get("project_id", "")
        
        project_name = project_spec.get("title", "GeneratedGame")
        build_target = "Android"  # Default to Android for mobile
        
        self.log_agent_action("build_game", "started",
                            project=project_name, project_id=project_id)
        
        try:
            build_path = await self._build_complete_game(
                project_id, project_name, build_target
            )
            
            if build_path:
                # Publish build completed event
                await publish_event(
                    EventType.BUILD_COMPLETED,
                    "builder_agent",
                    {
                        "project_id": project_id,
                        "project_name": project_name,
                        "build_target": build_target,
                        "build_path": str(build_path),
                        "success": True
                    }
                )
                
                return {
                    "success": True,
                    "build_path": str(build_path),
                    "build_target": build_target,
                    "project_name": project_name
                }
            else:
                await publish_event(
                    EventType.BUILD_COMPLETED,
                    "builder_agent",
                    {
                        "project_id": project_id,
                        "project_name": project_name,
                        "success": False,
                        "error": "Build failed"
                    }
                )
                
                return {
                    "success": False,
                    "error": "Build failed"
                }
                
        except Exception as e:
            self.log_agent_error("build_game", e,
                               project=project_name, project_id=project_id)
            
            await publish_event(
                EventType.BUILD_COMPLETED,
                "builder_agent",
                {
                    "project_id": project_id,
                    "project_name": project_name,
                    "success": False,
                    "error": str(e)
                }
            )
            raise
            
    async def _build_complete_game(
        self,
        project_id: str,
        project_name: str,
        build_target: str
    ) -> Optional[Path]:
        """Build a complete game from generated assets."""
        
        # Create Unity project
        project_path = settings.unity_project_path / f"{project_name}_{project_id}"
        
        success = await self.unity_builder.create_project(project_name, project_path)
        if not success:
            return None
            
        self.active_projects[project_id] = project_path
        
        # Import all generated assets
        await self._import_project_assets(project_id, ["code", "sprites", "audio", "levels"])
        
        # Configure project settings
        await self._configure_project_settings(project_path, project_name)
        
        # Build the project
        build_path = await self.unity_builder.build_project(
            project_path, build_target
        )
        
        return build_path
        
    async def _import_project_assets(
        self,
        project_id: str,
        asset_types: List[str]
    ) -> int:
        """Import generated assets into Unity project."""
        
        project_path = self.active_projects.get(project_id)
        if not project_path:
            raise ValueError(f"No active project found for ID: {project_id}")
            
        imported_count = 0
        
        for asset_type in asset_types:
            assets = self.asset_manager.list_project_assets(project_id, asset_type)
            
            for asset in assets:
                # Download asset to temp location
                temp_path = settings.data_dir / "temp" / asset.filename
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                
                success = await self.asset_manager.download_asset(
                    asset.id, temp_path
                )
                
                if success:
                    # Determine Unity destination path
                    unity_dest = self._get_unity_destination(asset_type, asset.filename)
                    
                    # Import to Unity project
                    await self.unity_builder.import_assets(
                        project_path,
                        [(temp_path, unity_dest)]
                    )
                    
                    imported_count += 1
                    
                    # Clean up temp file
                    temp_path.unlink(missing_ok=True)
                    
        self.logger.info(
            "Imported assets to Unity project",
            project_id=project_id,
            count=imported_count
        )
        
        return imported_count
        
    def _get_unity_destination(self, asset_type: str, filename: str) -> str:
        """Get Unity destination path for asset type."""
        
        destinations = {
            "code": f"Scripts/{filename}",
            "sprites": f"Sprites/{filename}",
            "character_sprites": f"Sprites/Characters/{filename}",
            "background": f"Sprites/Backgrounds/{filename}",
            "ui_element": f"Sprites/UI/{filename}",
            "audio": f"Audio/{filename}",
            "bgm": f"Audio/BGM/{filename}",
            "sfx": f"Audio/SFX/{filename}",
            "levels": f"Resources/Levels/{filename}",
        }
        
        return destinations.get(asset_type, f"Generated/{filename}")
        
    async def _configure_project_settings(
        self,
        project_path: Path,
        project_name: str
    ) -> None:
        """Configure Unity project settings for mobile deployment."""
        
        # This would normally involve modifying Unity's ProjectSettings files
        # For now, we'll create a simple configuration script
        
        config_script = f"""
using UnityEngine;

public class ProjectConfig
{{
    [RuntimeInitializeOnLoadMethod]
    static void Initialize()
    {{
        Application.targetFrameRate = 60;
        QualitySettings.vSyncCount = 1;
        
        // Mobile optimizations
        QualitySettings.pixelLightCount = 1;
        QualitySettings.shadows = ShadowQuality.Disable;
        QualitySettings.shadowResolution = ShadowResolution.Low;
        
        Debug.Log("Project {project_name} initialized");
    }}
}}
"""
        
        config_path = project_path / "Assets" / "Scripts" / "ProjectConfig.cs"
        config_path.write_text(config_script)
        
    async def _create_unity_project(
        self,
        project_name: str,
        template: str
    ) -> bool:
        """Create a new Unity project."""
        
        project_path = settings.unity_project_path / project_name
        return await self.unity_builder.create_project(project_name, project_path)
        
    async def _handle_create_project(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Unity project creation."""
        
        project_name = params.get("project_name", "NewProject")
        template = params.get("template", "2d")
        
        success = await self._create_unity_project(project_name, template)
        
        return {
            "success": success,
            "project_name": project_name,
            "template": template
        }
        
    async def _handle_import_assets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle asset importing."""
        
        project_id = params.get("project_id", "")
        asset_types = params.get("asset_types", ["code", "sprites", "audio"])
        
        count = await self._import_project_assets(project_id, asset_types)
        
        return {
            "imported_count": count,
            "project_id": project_id,
            "asset_types": asset_types
        }