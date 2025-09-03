"""Codex Agent - specialized in generating C# game code and Unity scripts."""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field

from ..core.agent_framework import BaseAgent, AgentType, AgentTask
from ..core.message_queue import EventType, publish_event
from ..core.logging import get_logger


class CodeGenRequest(BaseModel):
    """Request for code generation."""
    project_id: str = Field(description="Project identifier")
    component_type: str = Field(description="Type of component to generate")
    specifications: Dict[str, Any] = Field(description="Component specifications")
    dependencies: List[str] = Field(default=[], description="Required dependencies")


class GeneratedCode(BaseModel):
    """Generated code structure."""
    filename: str = Field(description="Name of the generated file")
    content: str = Field(description="Generated code content")
    namespace: str = Field(description="Code namespace")
    dependencies: List[str] = Field(description="Required dependencies")
    description: str = Field(description="Description of the code")


class CodexTools:
    """Tools available to the Codex Agent."""
    
    @staticmethod
    def create_generate_player_controller_tool(agent: "CodexAgent") -> BaseTool:
        """Tool for generating player controller scripts."""
        
        class PlayerControllerInput(BaseModel):
            movement_type: str = Field(description="Type of movement (2d_platformer, top_down, side_scroller)")
            controls: Dict[str, str] = Field(description="Control scheme mapping")
            physics_type: str = Field(description="Physics type (rigidbody, transform, character_controller)")
            
        async def generate_player_controller(
            movement_type: str,
            controls: Dict[str, str],
            physics_type: str
        ) -> str:
            try:
                code = await agent._generate_player_controller(
                    movement_type, controls, physics_type
                )
                return f"Generated PlayerController.cs with {movement_type} movement"
            except Exception as e:
                return f"Error generating player controller: {str(e)}"
                
        return StructuredTool.from_function(
            func=generate_player_controller,
            name="generate_player_controller",
            description="Generate a Unity player controller script",
            args_schema=PlayerControllerInput
        )
        
    @staticmethod
    def create_generate_game_manager_tool(agent: "CodexAgent") -> BaseTool:
        """Tool for generating game manager scripts."""
        
        class GameManagerInput(BaseModel):
            game_mechanics: List[str] = Field(description="List of game mechanics to manage")
            ui_elements: List[str] = Field(description="UI elements to control")
            
        async def generate_game_manager(
            game_mechanics: List[str],
            ui_elements: List[str]
        ) -> str:
            try:
                code = await agent._generate_game_manager(game_mechanics, ui_elements)
                return f"Generated GameManager.cs with {len(game_mechanics)} mechanics"
            except Exception as e:
                return f"Error generating game manager: {str(e)}"
                
        return StructuredTool.from_function(
            func=generate_game_manager,
            name="generate_game_manager",
            description="Generate a Unity game manager script",
            args_schema=GameManagerInput
        )
        
    @staticmethod
    def create_generate_ui_controller_tool(agent: "CodexAgent") -> BaseTool:
        """Tool for generating UI controller scripts."""
        
        class UIControllerInput(BaseModel):
            ui_type: str = Field(description="Type of UI (menu, hud, inventory, settings)")
            elements: List[Dict[str, str]] = Field(description="UI elements and their types")
            
        async def generate_ui_controller(
            ui_type: str,
            elements: List[Dict[str, str]]
        ) -> str:
            try:
                code = await agent._generate_ui_controller(ui_type, elements)
                return f"Generated {ui_type}UI.cs controller"
            except Exception as e:
                return f"Error generating UI controller: {str(e)}"
                
        return StructuredTool.from_function(
            func=generate_ui_controller,
            name="generate_ui_controller", 
            description="Generate a Unity UI controller script",
            args_schema=UIControllerInput
        )


class CodexAgent(BaseAgent):
    """Specialist agent for generating C# Unity game code."""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="codex_agent",
            agent_type=AgentType.CODEX,
            **kwargs
        )
        
        self.generated_code: Dict[str, GeneratedCode] = {}
        
        # Create tools
        self.tools = [
            CodexTools.create_generate_player_controller_tool(self),
            CodexTools.create_generate_game_manager_tool(self),
            CodexTools.create_generate_ui_controller_tool(self),
        ]
        
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the LangChain agent executor for code generation."""
        
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
        """Get the system prompt for the Codex agent."""
        return """
        You are the Codex Agent, a specialist in generating high-quality C# code for Unity game development.
        
        Your expertise includes:
        - Unity C# scripting and best practices
        - Game programming patterns (Singleton, Observer, State Machine, Object Pooling)
        - Player controllers and character movement
        - Game mechanics implementation
        - UI systems and event handling
        - Audio management and sound systems
        - Performance optimization techniques
        
        Unity-specific knowledge:
        - MonoBehaviour lifecycle (Awake, Start, Update, FixedUpdate)
        - Component-based architecture
        - Unity API usage (Transform, Rigidbody, Collider, etc.)
        - Coroutines and async programming
        - ScriptableObjects for data management
        - Unity Events and UnityActions
        
        Code quality standards:
        - Clean, readable, and well-documented code
        - Proper error handling and null checks
        - Performance-conscious implementations
        - Consistent naming conventions (PascalCase for public, camelCase for private)
        - Appropriate use of Unity attributes [SerializeField], [RequireComponent], etc.
        
        Always generate production-ready code that follows Unity best practices and is optimized for mobile platforms.
        """
        
    async def execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute a code generation task."""
        
        action = task.action
        params = task.parameters
        
        if action == "generate_gameplay_code":
            return await self._generate_gameplay_code(params)
        elif action == "generate_ui_code":
            return await self._generate_ui_code(params)
        elif action == "generate_manager_code":
            return await self._generate_manager_code(params)
        else:
            # Use agent executor for complex code generation
            result = await self.agent_executor.ainvoke({
                "input": f"Generate code for: {action} with requirements: {params}"
            })
            return {"result": result["output"]}
            
    async def _generate_gameplay_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate gameplay mechanic code."""
        
        mechanic = params.get("mechanic", "")
        project_spec = params.get("project_spec", {})
        project_id = params.get("project_id", "")
        
        self.log_agent_action("generate_gameplay_code", "started",
                            mechanic=mechanic, project_id=project_id)
        
        try:
            # Generate code based on mechanic type
            if "jump" in mechanic.lower() or "platform" in mechanic.lower():
                code = await self._generate_platformer_code(project_spec)
            elif "shoot" in mechanic.lower() or "bullet" in mechanic.lower():
                code = await self._generate_shooting_code(project_spec)
            elif "collect" in mechanic.lower() or "pickup" in mechanic.lower():
                code = await self._generate_collection_code(project_spec)
            else:
                code = await self._generate_generic_mechanic_code(mechanic, project_spec)
                
            # Store generated code
            self.generated_code[f"{project_id}_{mechanic}"] = code
            
            # Publish asset generated event
            await publish_event(
                EventType.ASSET_GENERATED,
                "codex_agent",
                {
                    "asset_type": "code",
                    "filename": code.filename,
                    "project_id": project_id,
                    "mechanic": mechanic
                }
            )
            
            return {
                "filename": code.filename,
                "lines_of_code": len(code.content.split('\n')),
                "dependencies": code.dependencies,
                "description": code.description
            }
            
        except Exception as e:
            self.log_agent_error("generate_gameplay_code", e,
                               mechanic=mechanic, project_id=project_id)
            raise
            
    async def _generate_platformer_code(self, project_spec: Dict[str, Any]) -> GeneratedCode:
        """Generate 2D platformer player controller."""
        
        code_template = '''using UnityEngine;

[RequireComponent(typeof(Rigidbody2D))]
[RequireComponent(typeof(Collider2D))]
public class PlayerController : MonoBehaviour
{
    [Header("Movement Settings")]
    [SerializeField] private float moveSpeed = 5f;
    [SerializeField] private float jumpForce = 10f;
    [SerializeField] private LayerMask groundLayerMask = 1;
    
    [Header("Ground Check")]
    [SerializeField] private Transform groundCheck;
    [SerializeField] private float groundCheckRadius = 0.2f;
    
    // Components
    private Rigidbody2D rb2d;
    private Animator animator;
    private SpriteRenderer spriteRenderer;
    
    // Input
    private float horizontalInput;
    private bool jumpInput;
    
    // State
    private bool isGrounded;
    private bool facingRight = true;
    
    void Awake()
    {
        rb2d = GetComponent<Rigidbody2D>();
        animator = GetComponent<Animator>();
        spriteRenderer = GetComponent<SpriteRenderer>();
    }
    
    void Update()
    {
        HandleInput();
        CheckGrounded();
        UpdateAnimations();
    }
    
    void FixedUpdate()
    {
        HandleMovement();
    }
    
    private void HandleInput()
    {
        horizontalInput = Input.GetAxis("Horizontal");
        jumpInput = Input.GetButtonDown("Jump");
    }
    
    private void HandleMovement()
    {
        // Horizontal movement
        rb2d.velocity = new Vector2(horizontalInput * moveSpeed, rb2d.velocity.y);
        
        // Jumping
        if (jumpInput && isGrounded)
        {
            rb2d.velocity = new Vector2(rb2d.velocity.x, jumpForce);
        }
        
        // Sprite flipping
        if (horizontalInput > 0 && !facingRight)
        {
            Flip();
        }
        else if (horizontalInput < 0 && facingRight)
        {
            Flip();
        }
    }
    
    private void CheckGrounded()
    {
        isGrounded = Physics2D.OverlapCircle(groundCheck.position, groundCheckRadius, groundLayerMask);
    }
    
    private void UpdateAnimations()
    {
        if (animator != null)
        {
            animator.SetFloat("Speed", Mathf.Abs(horizontalInput));
            animator.SetBool("IsGrounded", isGrounded);
            animator.SetFloat("VerticalSpeed", rb2d.velocity.y);
        }
    }
    
    private void Flip()
    {
        facingRight = !facingRight;
        Vector3 scale = transform.localScale;
        scale.x *= -1;
        transform.localScale = scale;
    }
    
    void OnDrawGizmosSelected()
    {
        if (groundCheck != null)
        {
            Gizmos.color = isGrounded ? Color.green : Color.red;
            Gizmos.DrawWireSphere(groundCheck.position, groundCheckRadius);
        }
    }
}'''
        
        return GeneratedCode(
            filename="PlayerController.cs",
            content=code_template,
            namespace="GameCore",
            dependencies=["UnityEngine"],
            description="2D platformer player controller with jumping and movement"
        )
        
    async def _generate_shooting_code(self, project_spec: Dict[str, Any]) -> GeneratedCode:
        """Generate shooting mechanism code."""
        
        code_template = '''using UnityEngine;

public class ShootingController : MonoBehaviour
{
    [Header("Shooting Settings")]
    [SerializeField] private GameObject bulletPrefab;
    [SerializeField] private Transform firePoint;
    [SerializeField] private float bulletSpeed = 10f;
    [SerializeField] private float fireRate = 0.5f;
    
    [Header("Audio")]
    [SerializeField] private AudioClip shootSound;
    
    // Components
    private AudioSource audioSource;
    
    // State
    private float nextFireTime = 0f;
    
    void Awake()
    {
        audioSource = GetComponent<AudioSource>();
        if (audioSource == null)
        {
            audioSource = gameObject.AddComponent<AudioSource>();
        }
    }
    
    void Update()
    {
        HandleShootInput();
    }
    
    private void HandleShootInput()
    {
        if (Input.GetButton("Fire1") && Time.time >= nextFireTime)
        {
            Shoot();
            nextFireTime = Time.time + fireRate;
        }
    }
    
    public void Shoot()
    {
        if (bulletPrefab != null && firePoint != null)
        {
            // Create bullet
            GameObject bullet = Instantiate(bulletPrefab, firePoint.position, firePoint.rotation);
            
            // Set bullet velocity
            Rigidbody2D bulletRb = bullet.GetComponent<Rigidbody2D>();
            if (bulletRb != null)
            {
                bulletRb.velocity = firePoint.right * bulletSpeed;
            }
            
            // Play sound
            if (shootSound != null && audioSource != null)
            {
                audioSource.PlayOneShot(shootSound);
            }
            
            // Destroy bullet after 5 seconds
            Destroy(bullet, 5f);
        }
    }
}'''
        
        return GeneratedCode(
            filename="ShootingController.cs",
            content=code_template,
            namespace="GameCore",
            dependencies=["UnityEngine"],
            description="Shooting controller with bullet spawning and audio"
        )
        
    async def _generate_collection_code(self, project_spec: Dict[str, Any]) -> GeneratedCode:
        """Generate collectible item code."""
        
        code_template = '''using UnityEngine;

public class Collectible : MonoBehaviour
{
    [Header("Collectible Settings")]
    [SerializeField] private int points = 10;
    [SerializeField] private AudioClip collectSound;
    [SerializeField] private GameObject collectEffect;
    
    [Header("Animation")]
    [SerializeField] private float rotationSpeed = 90f;
    [SerializeField] private float bobSpeed = 2f;
    [SerializeField] private float bobHeight = 0.5f;
    
    // Private variables
    private Vector3 startPosition;
    private AudioSource audioSource;
    
    void Start()
    {
        startPosition = transform.position;
        audioSource = Camera.main.GetComponent<AudioSource>();
        
        if (audioSource == null)
        {
            audioSource = FindObjectOfType<AudioSource>();
        }
    }
    
    void Update()
    {
        // Rotate the collectible
        transform.Rotate(Vector3.up, rotationSpeed * Time.deltaTime);
        
        // Bob up and down
        float newY = startPosition.y + Mathf.Sin(Time.time * bobSpeed) * bobHeight;
        transform.position = new Vector3(transform.position.x, newY, transform.position.z);
    }
    
    void OnTriggerEnter2D(Collider2D other)
    {
        if (other.CompareTag("Player"))
        {
            Collect();
        }
    }
    
    private void Collect()
    {
        // Add points to game manager
        GameManager gameManager = FindObjectOfType<GameManager>();
        if (gameManager != null)
        {
            gameManager.AddPoints(points);
        }
        
        // Play collection sound
        if (collectSound != null && audioSource != null)
        {
            audioSource.PlayOneShot(collectSound);
        }
        
        // Spawn collection effect
        if (collectEffect != null)
        {
            Instantiate(collectEffect, transform.position, Quaternion.identity);
        }
        
        // Destroy the collectible
        Destroy(gameObject);
    }
}'''
        
        return GeneratedCode(
            filename="Collectible.cs",
            content=code_template,
            namespace="GameCore",
            dependencies=["UnityEngine"],
            description="Collectible item with animation and point scoring"
        )
        
    async def _generate_generic_mechanic_code(self, mechanic: str, project_spec: Dict[str, Any]) -> GeneratedCode:
        """Generate code for generic game mechanics using LLM."""
        
        prompt = f"""
        Generate a Unity C# script for the game mechanic: "{mechanic}"
        
        Project context: {json.dumps(project_spec, indent=2)}
        
        Requirements:
        - Follow Unity best practices and coding standards
        - Include proper error handling and null checks
        - Use appropriate Unity attributes and components
        - Optimize for mobile performance
        - Include clear documentation
        - Make it production-ready
        
        Generate only the C# code, no additional explanation.
        """
        
        response = await self.llm.ainvoke(prompt)
        code_content = response.content.strip()
        
        # Extract class name for filename
        class_name = "GameMechanic"
        lines = code_content.split('\n')
        for line in lines:
            if 'public class' in line:
                parts = line.split('public class')[1].split()
                if parts:
                    class_name = parts[0].split(':')[0].strip()
                    break
                    
        return GeneratedCode(
            filename=f"{class_name}.cs",
            content=code_content,
            namespace="GameCore",
            dependencies=["UnityEngine"],
            description=f"Generated code for {mechanic} mechanic"
        )
        
    async def _generate_player_controller(
        self, 
        movement_type: str,
        controls: Dict[str, str],
        physics_type: str
    ) -> GeneratedCode:
        """Generate a player controller based on specifications."""
        
        if movement_type == "2d_platformer":
            return await self._generate_platformer_code({})
        else:
            # Use LLM for other movement types
            return await self._generate_generic_mechanic_code(
                f"{movement_type} player controller", {}
            )
            
    async def _generate_game_manager(
        self, 
        game_mechanics: List[str],
        ui_elements: List[str]
    ) -> GeneratedCode:
        """Generate a game manager script."""
        
        code_template = '''using UnityEngine;
using UnityEngine.SceneManagement;

public class GameManager : MonoBehaviour
{
    [Header("Game State")]
    [SerializeField] private int score = 0;
    [SerializeField] private int lives = 3;
    [SerializeField] private bool gameOver = false;
    
    [Header("UI References")]
    [SerializeField] private TMPro.TextMeshProUGUI scoreText;
    [SerializeField] private TMPro.TextMeshProUGUI livesText;
    [SerializeField] private GameObject gameOverPanel;
    
    // Singleton instance
    public static GameManager Instance { get; private set; }
    
    // Events
    public System.Action<int> OnScoreChanged;
    public System.Action<int> OnLivesChanged;
    public System.Action OnGameOver;
    
    void Awake()
    {
        // Singleton pattern
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }
    
    void Start()
    {
        UpdateUI();
    }
    
    public void AddPoints(int points)
    {
        score += points;
        OnScoreChanged?.Invoke(score);
        UpdateUI();
    }
    
    public void LoseLife()
    {
        if (!gameOver)
        {
            lives--;
            OnLivesChanged?.Invoke(lives);
            
            if (lives <= 0)
            {
                GameOver();
            }
            
            UpdateUI();
        }
    }
    
    public void GameOver()
    {
        gameOver = true;
        OnGameOver?.Invoke();
        
        if (gameOverPanel != null)
        {
            gameOverPanel.SetActive(true);
        }
        
        Time.timeScale = 0f;
    }
    
    public void RestartGame()
    {
        Time.timeScale = 1f;
        SceneManager.LoadScene(SceneManager.GetActiveScene().buildIndex);
    }
    
    private void UpdateUI()
    {
        if (scoreText != null)
            scoreText.text = "Score: " + score;
            
        if (livesText != null)
            livesText.text = "Lives: " + lives;
    }
    
    // Getters
    public int Score => score;
    public int Lives => lives;
    public bool IsGameOver => gameOver;
}'''
        
        return GeneratedCode(
            filename="GameManager.cs",
            content=code_template,
            namespace="GameCore", 
            dependencies=["UnityEngine", "UnityEngine.SceneManagement"],
            description="Game manager with score, lives, and game state management"
        )
        
    async def _generate_ui_controller(
        self,
        ui_type: str,
        elements: List[Dict[str, str]]
    ) -> GeneratedCode:
        """Generate a UI controller script."""
        
        class_name = f"{ui_type.capitalize()}UI"
        
        code_template = f'''using UnityEngine;
using UnityEngine.UI;

public class {class_name} : MonoBehaviour
{{
    [Header("UI Elements")]'''
    
        # Add serialized fields for UI elements
        for element in elements:
            element_name = element.get('name', 'element')
            element_type = element.get('type', 'Button')
            code_template += f'\n    [SerializeField] private {element_type} {element_name};'
            
        code_template += '''
    
    void Start()
    {
        InitializeUI();
    }
    
    private void InitializeUI()
    {
        // Initialize UI elements and add event listeners
        SetupEventListeners();
    }
    
    private void SetupEventListeners()
    {
        // Add button click listeners and other UI events here
    }
    
    public void Show()
    {
        gameObject.SetActive(true);
    }
    
    public void Hide()
    {
        gameObject.SetActive(false);
    }
}'''
        
        return GeneratedCode(
            filename=f"{class_name}.cs",
            content=code_template,
            namespace="GameUI",
            dependencies=["UnityEngine", "UnityEngine.UI"],
            description=f"UI controller for {ui_type} interface"
        )
        
    async def _generate_ui_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate UI-related code."""
        ui_type = params.get("ui_type", "menu")
        elements = params.get("elements", [])
        
        code = await self._generate_ui_controller(ui_type, elements)
        
        return {
            "filename": code.filename,
            "description": code.description,
            "elements_count": len(elements)
        }
        
    async def _generate_manager_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate manager-related code."""
        mechanics = params.get("mechanics", [])
        ui_elements = params.get("ui_elements", [])
        
        code = await self._generate_game_manager(mechanics, ui_elements)
        
        return {
            "filename": code.filename,
            "description": code.description,
            "mechanics_count": len(mechanics)
        }