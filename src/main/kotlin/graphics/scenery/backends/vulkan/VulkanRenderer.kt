package graphics.scenery.backends.vulkan

import cleargl.GLMatrix
import cleargl.GLVector
import graphics.scenery.*
import org.lwjgl.PointerBuffer
import org.lwjgl.glfw.GLFW.*
import org.lwjgl.glfw.GLFWVulkan.*
import org.lwjgl.glfw.GLFWWindowSizeCallback
import org.lwjgl.system.MemoryUtil.*
import org.lwjgl.vulkan.*
import org.lwjgl.vulkan.EXTDebugReport.*
import org.lwjgl.vulkan.KHRSurface.*
import org.lwjgl.vulkan.KHRSwapchain.*
import org.lwjgl.vulkan.VK10.*
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import graphics.scenery.backends.RenderConfigReader
import graphics.scenery.backends.Renderer
import graphics.scenery.backends.SceneryWindow
import graphics.scenery.backends.createRenderpassFlow
import graphics.scenery.backends.ShaderPreference
import graphics.scenery.fonts.SDFFontAtlas
import graphics.scenery.utils.Statistics
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte
import java.io.File
import java.nio.ByteBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import javax.imageio.ImageIO
import kotlin.concurrent.thread


/**
 * <Description>
 *
 * @author Ulrik Günther <hello@ulrik.is>
 */
open class VulkanRenderer(applicationName: String,
                          scene: Scene,
                          windowWidth: Int,
                          windowHeight: Int,
                          renderConfigFile: String = System.getProperty("scenery.Renderer.Config", "DeferredShading.yml")) : Renderer, AutoCloseable {

    // helper classes
    data class PresentHelpers(
        var signalSemaphore: LongBuffer = memAllocLong(1),
        var waitSemaphore: LongBuffer = memAllocLong(1),
        var commandBuffers: PointerBuffer = memAllocPointer(1),
        var waitStages: IntBuffer = memAllocInt(2)
    )

    enum class VertexDataKinds {
        coords_none,
        coords_normals_texcoords,
        coords_texcoords,
        coords_normals
    }

    enum class StandardSemaphores {
        render_complete,
        image_available,
        present_complete
    }

    data class VertexDescription(
        var state: VkPipelineVertexInputStateCreateInfo,
        var attributeDescription: VkVertexInputAttributeDescription.Buffer?,
        var bindingDescription: VkVertexInputBindingDescription.Buffer?
    )

    data class CommandPools(
        var Standard: Long = -1L,
        var Render: Long = -1L,
        var Compute: Long = -1L
    )

    class DeviceAndGraphicsQueueFamily {
        internal var device: VkDevice? = null
        internal var queueFamilyIndex: Int = 0
        internal var memoryProperties: VkPhysicalDeviceMemoryProperties? = null
    }

    class ColorFormatAndSpace {
        internal var colorFormat: Int = 0
        internal var colorSpace: Int = 0
    }

    inner class Swapchain : AutoCloseable {
        internal var handle: Long = 0
        internal var images: LongArray? = null
        internal var imageViews: LongArray? = null

        override fun close() {
//            imageViews?.map { vkDestroyImageView(device, it, null) }
//            images?.map { vkDestroyImage(device, it, null) }
        }
    }

    class Pipeline {
        internal var pipeline: Long = 0
        internal var layout: Long = 0
    }

    inner class SwapchainRecreator {
        var mustRecreate = true

        fun recreate() {
            logger.info("Recreating Swapchain at frame $frames")
            // create new swapchain with changed surface parameters
            with(VU.newCommandBuffer(device, commandPools.Standard, autostart = true)) {
                val oldChain = swapchain?.handle ?: VK_NULL_HANDLE

                // Create the swapchain (this will also add a memory barrier to initialize the framebuffer images)
                swapchain = createSwapChain(
                    device, physicalDevice,
                    surface, oldChain,
                    window.width, window.height,
                    colorFormatAndSpace.colorFormat,
                    colorFormatAndSpace.colorSpace)

                this.endCommandBuffer(device, commandPools.Standard, queue, flush = true, dealloc = true)

                this
            }

            val pipelineCacheInfo = VkPipelineCacheCreateInfo.calloc()
                .sType(VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO)
                .pNext(NULL)
                .flags(VK_FLAGS_NONE)

            val refreshResolutionDependentResources = {
                pipelineCache = VU.run(memAllocLong(1), "create pipeline cache",
                    { vkCreatePipelineCache(device, pipelineCacheInfo, null, this) },
                    { pipelineCacheInfo.free() })

                renderpasses.forEach { s, vulkanRenderpass ->
                    vulkanRenderpass.close()
                }

                renderpasses.clear()

                renderpasses = prepareRenderpassesFromConfig(renderConfig, window.width, window.height)

                semaphores.forEach { it.value.forEach { semaphore -> vkDestroySemaphore(device, semaphore, null) } }
                semaphores = prepareStandardSemaphores(device)

                // Create render command buffers
                if (renderCommandBuffers != null) {
                    vkResetCommandPool(device, commandPools.Render, VK_FLAGS_NONE)
                }
            }

            refreshResolutionDependentResources.invoke()

            totalFrames = 0
            mustRecreate = false
        }
    }

    var debugCallback = object : VkDebugReportCallbackEXT() {
        override operator fun invoke(flags: Int, objectType: Int, obj: Long, location: Long, messageCode: Int, pLayerPrefix: Long, pMessage: Long, pUserData: Long): Int {
            var dbg = if (flags and VK_DEBUG_REPORT_DEBUG_BIT_EXT == 1) {
                " (debug)"
            } else {
                ""
            }

            if (flags and VK_DEBUG_REPORT_ERROR_BIT_EXT == 0) {
                logger.error("!! Validation$dbg: " + getString(pMessage))
            } else if (flags and VK_DEBUG_REPORT_WARNING_BIT_EXT == 0) {
                logger.warn("!! Validation$dbg: " + getString(pMessage))
            } else if (flags and VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT == 0) {
                logger.error("!! Validation (performance)$dbg: " + getString(pMessage))
            } else if (flags and VK_DEBUG_REPORT_INFORMATION_BIT_EXT == 0) {
                logger.info("!! Validation$dbg: " + getString(pMessage))
            } else {
                logger.info("!! Validation (unknown message type)$dbg: " + getString(pMessage))
            }

            return VK_FALSE
        }
    }

    // helper classes end


    // helper vars
    private val VK_FLAGS_NONE: Int = 0
    private var MAX_TEXTURES = 2048 * 16
    private var MAX_UBOS = 2048
    private var MAX_INPUT_ATTACHMENTS = 32
    private val UINT64_MAX: Long = -1L
    private val WINDOW_RESIZE_TIMEOUT = 200 * 10e6


    private val MATERIAL_HAS_DIFFUSE = 0x0001
    private val MATERIAL_HAS_AMBIENT = 0x0002
    private val MATERIAL_HAS_SPECULAR = 0x0004
    private val MATERIAL_HAS_NORMAL = 0x0008

    // end helper vars

    override var hub: Hub? = null
    protected var applicationName = ""
    override var settings: Settings = Settings()
    protected var logger: Logger = LoggerFactory.getLogger("VulkanRenderer")
    override var shouldClose = false
    var toggleFullscreen = false
    override var managesRenderLoop = false
    var screenshotRequested = false

    var firstWaitSemaphore = memAllocLong(1)

    var scene: Scene = Scene()

    protected var commandPools = CommandPools()
    protected var renderpasses = LinkedHashMap<String, VulkanRenderpass>()
    /** Cache for [SDFFontAtlas]es used for font rendering */
    protected var fontAtlas = HashMap<String, SDFFontAtlas>()

    protected var renderCommandBuffers: Array<VkCommandBuffer>? = null

    protected val validation = java.lang.Boolean.parseBoolean(System.getProperty("scenery.VulkanRenderer.EnableValidation", "false"))
    protected val layers = arrayOf<ByteBuffer>(memUTF8("VK_LAYER_LUNARG_standard_validation"))

    protected var instance: VkInstance

    protected var debugCallbackHandle: Long
    protected var windowSizeCallback: GLFWWindowSizeCallback
    protected var physicalDevice: VkPhysicalDevice
    protected var deviceAndGraphicsQueueFamily: DeviceAndGraphicsQueueFamily
    protected var device: VkDevice
    protected var queueFamilyIndex: Int
    protected var memoryProperties: VkPhysicalDeviceMemoryProperties

    protected var surface: Long

    protected var semaphoreCreateInfo: VkSemaphoreCreateInfo

    // Create static Vulkan resources
    protected var colorFormatAndSpace: ColorFormatAndSpace
    protected var postPresentCommandBuffer: VkCommandBuffer
    protected var queue: VkQueue
    protected var descriptorPool: Long

    protected var standardUBOs = ConcurrentHashMap<String, UBO>()

    protected var swapchain: Swapchain? = null
    protected var pSwapchains: LongBuffer = memAllocLong(1)
    protected var swapchainImage: IntBuffer = memAllocInt(1)
    protected var ph = PresentHelpers()

    override var window = SceneryWindow()

    protected val swapchainRecreator: SwapchainRecreator
    protected var pipelineCache: Long = -1L
    protected var vertexDescriptors = ConcurrentHashMap<VertexDataKinds, VertexDescription>()
    protected var sceneUBOs = ConcurrentHashMap<Node, UBO>()
    protected var semaphores = ConcurrentHashMap<StandardSemaphores, Array<Long>>()
    protected var buffers = HashMap<String, VulkanBuffer>()
    protected var textureCache = ConcurrentHashMap<String, VulkanTexture>()
    protected var descriptorSetLayouts = ConcurrentHashMap<String, Long>()
    protected var descriptorSets = ConcurrentHashMap<String, Long>()

    protected var lastTime = System.nanoTime()
    protected var time = 0.0f
    protected var fps = 0
    protected var frames = 0
    protected var totalFrames = 0L
    protected var heartbeatTimer = Timer()
    protected var lastResize = -1L

    private var renderConfig: RenderConfigReader.RenderConfig
    private var flow: List<String> = listOf()

    var renderConfigFile = ""
        set(config) {
            field = config

            this.renderConfig = RenderConfigReader().loadFromFile(renderConfigFile)

            // check for null as this is used in the constructor as well where
            // the swapchain recreator is not yet initialized
            if(swapchainRecreator != null) {
                swapchainRecreator.mustRecreate = true
                logger.info("Loaded ${renderConfig.name} (${renderConfig.description ?: "no description"})")
            }
        }

    init {
        window.width = windowWidth
        window.height = windowHeight

        this.applicationName = applicationName
        this.scene = scene

        this.settings = getDefaultRendererSettings()

        logger.debug("Loading rendering config from $renderConfigFile")
        this.renderConfigFile = renderConfigFile
        this.renderConfig = RenderConfigReader().loadFromFile(renderConfigFile)

        logger.info("Loaded ${renderConfig.name} (${renderConfig.description ?: "no description"})")

        if (!glfwInit()) {
            throw RuntimeException("Failed to initialize GLFW")
        }
        if (!glfwVulkanSupported()) {
            throw AssertionError("Failed to find Vulkan loader. Do you have the most recent graphics drivers installed?")
        }

        /* Look for instance extensions */
        val requiredExtensions = glfwGetRequiredInstanceExtensions() ?: throw AssertionError("Failed to find list of required Vulkan extensions")

        // Create the Vulkan instance
        instance = createInstance(requiredExtensions)
        debugCallbackHandle = setupDebugging(instance,
            VK_DEBUG_REPORT_ERROR_BIT_EXT or VK_DEBUG_REPORT_WARNING_BIT_EXT,
            debugCallback)

        physicalDevice = getPhysicalDevice(instance)
        deviceAndGraphicsQueueFamily = createDeviceAndGetGraphicsQueueFamily(physicalDevice)
        device = deviceAndGraphicsQueueFamily.device!!
        queueFamilyIndex = deviceAndGraphicsQueueFamily.queueFamilyIndex
        memoryProperties = deviceAndGraphicsQueueFamily.memoryProperties!!

        // Create GLFW window
        glfwDefaultWindowHints()
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API)
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE)
        window.glfwWindow = glfwCreateWindow(windowWidth, windowHeight, "scenery", NULL, NULL)
        glfwSetWindowPos(window.glfwWindow!!, 100, 100)

        surface = VU.run(memAllocLong(1), "glfwCreateWindowSurface") {
            glfwCreateWindowSurface(instance, window.glfwWindow!!, null, this)
        }

        swapchainRecreator = SwapchainRecreator()


        // create resolution-independent resources
        colorFormatAndSpace = getColorFormatAndSpace(physicalDevice, surface)

        with(commandPools) {
            Render = createCommandPool(device, queueFamilyIndex)
            Standard = createCommandPool(device, queueFamilyIndex)
            Compute = createCommandPool(device, queueFamilyIndex)
        }

        postPresentCommandBuffer = VU.newCommandBuffer(device, commandPools.Standard)

        queue = VU.createDeviceQueue(device, queueFamilyIndex)

        descriptorPool = createDescriptorPool(device)
        vertexDescriptors = prepareStandardVertexDescriptors()

        buffers = prepareDefaultBuffers(device)
        descriptorSetLayouts = prepareDefaultDescriptorSetLayouts(device)
        standardUBOs = prepareDefaultUniformBuffers(device)

        prepareDescriptorSets(device, descriptorPool)
        prepareDefaultTextures(device)


        heartbeatTimer.scheduleAtFixedRate(object : TimerTask() {
            override fun run() {
                if(shouldClose) {
                    return
                }

                fps = frames
                frames = 0

                glfwSetWindowTitle(window.glfwWindow!!,
                    "$applicationName [${this@VulkanRenderer.javaClass.simpleName}, ${this@VulkanRenderer.renderConfig.name}${if (validation) {
                        " - VALIDATIONS ENABLED"
                    } else {
                        ""
                    }}] - $fps fps")
            }
        }, 0, 1000)

        // Handle canvas resize
        windowSizeCallback = object : GLFWWindowSizeCallback() {
            override operator fun invoke(glfwWindow: Long, w: Int, h: Int) {
                if (lastResize > 0L && lastResize + WINDOW_RESIZE_TIMEOUT < System.nanoTime()) {
                    lastResize = System.nanoTime()
                    return
                }

                if (window.width <= 0 || window.height <= 0)
                    return

                window.width = w
                window.height = h
                swapchainRecreator.mustRecreate = true
                lastResize = -1L
            }
        }

        glfwSetWindowSizeCallback(window.glfwWindow!!, windowSizeCallback)
        glfwShowWindow(window.glfwWindow!!)

        // Info struct to create a semaphore
        semaphoreCreateInfo = VkSemaphoreCreateInfo.calloc()
            .sType(VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO)
            .pNext(NULL)
            .flags(0)

        lastTime = System.nanoTime()
        time = 0.0f
    }

    /**
     * Returns the default [Settings] for [VulkanRenderer]
     *
     * Providing some sane defaults that may of course be overridden after
     * construction of the renderer.
     *
     * @return Default [Settings] values
     */
    protected fun getDefaultRendererSettings(): Settings {
        val ds = Settings()

        ds.set("wantsFullscreen", false)
        ds.set("isFullscreen", false)

        ds.set("ssao.Active", true)
        ds.set("ssao.FilterRadius", GLVector(0.0f, 0.0f))
        ds.set("ssao.DistanceThreshold", 50.0f)
        ds.set("ssao.Algorithm", 1)

        ds.set("vr.Active", false)
        ds.set("vr.DoAnaglyph", false)
        ds.set("vr.IPD", 0.0f)
        ds.set("vr.EyeDivisor", 1)

        ds.set("hdr.Active", true)
        ds.set("hdr.Exposure", 1.0f)
        ds.set("hdr.Gamma", 2.2f)

        ds.set("sdf.MaxDistance", 10)

        return ds
    }

    // source: http://stackoverflow.com/questions/34697828/parallel-operations-on-kotlin-collections
    // Thanks to Holger :-)
    fun <T, R> Iterable<T>.parallelMap(
        numThreads: Int = Runtime.getRuntime().availableProcessors(),
        exec: ExecutorService = Executors.newFixedThreadPool(numThreads),
        transform: (T) -> R): List<R> {

        // default size is just an inlined version of kotlin.collections.collectionSizeOrDefault
        val defaultSize = if (this is Collection<*>) this.size else 10
        val destination = Collections.synchronizedList(ArrayList<R>(defaultSize))

        for (item in this) {
            exec.submit { destination.add(transform(item)) }
        }

        exec.shutdown()
        exec.awaitTermination(1, TimeUnit.DAYS)

        return ArrayList<R>(destination)
    }

    fun setCurrentScene(scene: Scene) {
        this.scene = scene
    }

    /**
     * This function should initialize the scene contents.
     *
     * @param[scene] The scene to initialize.
     */
    override fun initializeScene() {
        logger.info("Scene initialization started.")

        this.scene.discover(this.scene, { it is HasGeometry })
//            .parallelMap(numThreads = System.getProperty("scenery.MaxInitThreads", "1").toInt()) { node ->
            .map { node ->
                logger.debug("Initializing object '${node.name}'")
                node.metadata.put("VulkanRenderer", VulkanObjectState())

                if (node is FontBoard) {
                    updateFontBoard(node)
                } else {
                    initializeNode(node)
                }
            }

        scene.initialized = true
        logger.info("Scene initialization complete.")
    }

    protected fun updateFontBoard(board: FontBoard) {
        val atlas = fontAtlas.getOrPut(board.fontFamily,
            { SDFFontAtlas(this.hub!!, board.fontFamily, maxDistance = settings.get<Int>("sdf.MaxDistance")) })
        val m = atlas.createMeshForString(board.text)

        board.vertices = m.vertices
        board.normals = m.normals
        board.indices = m.indices
        board.texcoords = m.texcoords

        if(board.initialized == false) {
            board.metadata.put("VulkanRenderer", VulkanObjectState())
            initializeNode(board)
        } else {
            updateNodeGeometry(board)
        }

        val s = board.metadata.get("VulkanRenderer") as VulkanObjectState

        val texture = textureCache.getOrPut("sdf-${board.fontFamily}", {
            val t = VulkanTexture(device, physicalDevice, memoryProperties,
                commandPools.Standard, queue,
                atlas.atlasWidth, atlas.atlasHeight, 1,
                format = VK_FORMAT_R32_SFLOAT,
                mipLevels = 3)

            t.copyFrom(atlas.getAtlas())
            t
        })

        s.textures.put("ambient", texture)
        s.textures.put("diffuse", texture)

        s.texturesToDescriptorSet(device, descriptorSetLayouts["ObjectTextures"]!!,
            descriptorPool,
            targetBinding = 0)

        board.dirty = false
        board.initialized = true
    }

    fun Boolean.toInt(): Int {
        return if (this) {
            1
        } else {
            0
        }
    }

    fun updateNodeGeometry(node: Node) {
        if(node is HasGeometry) {
            val s = node.metadata["VulkanRenderer"]!! as VulkanObjectState
            s.vertexBuffers.forEach {
                it.value.close() }

            createVertexBuffers(device, node, s)
        }
    }

    /**
     *
     */
    fun initializeNode(node: Node): Boolean {
        var s: VulkanObjectState

        s = node.metadata["VulkanRenderer"] as VulkanObjectState

        if (s.initialized) return true

        logger.debug("Initializing ${node.name} (${(node as HasGeometry).vertices.remaining() / node.vertexSize} vertices/${node.indices.remaining()} indices)")

        // determine vertex input type
        if (node.vertices.remaining() > 0 && node.normals.remaining() > 0 && node.texcoords.remaining() > 0) {
            s.vertexInputType = VertexDataKinds.coords_normals_texcoords
        }

        if (node.vertices.remaining() > 0 && node.normals.remaining() > 0 && node.texcoords.remaining() == 0) {
            s.vertexInputType = VertexDataKinds.coords_normals
        }

        if (node.vertices.remaining() > 0 && node.normals.remaining() == 0 && node.texcoords.remaining() > 0) {
            s.vertexInputType = VertexDataKinds.coords_texcoords
        }

        // create custom vertex description if necessary, else use one of the defaults
        s.vertexDescription = if (node.instanceMaster) {
            vertexDescriptionFromInstancedNode(node, vertexDescriptors[VertexDataKinds.coords_normals_texcoords]!!)
        } else {
            vertexDescriptors[s.vertexInputType]!!
        }

        if (node.instanceOf != null) {
            val parentMetadata = node.instanceOf!!.metadata["VulkanRenderer"] as VulkanObjectState

            if (!parentMetadata.initialized) {
                logger.info("Instance parent ${node.instanceOf!!} is not initialized yet, initializing now...")
                initializeNode(node.instanceOf!!)
            }

            if (!parentMetadata.vertexBuffers.containsKey("instance")) {
                createInstanceBuffer(device, node.instanceOf!!, parentMetadata)
            }

            return true
        }

        if (node.vertices.remaining() > 0) {
            s = createVertexBuffers(device, node, s)
        }

        val matricesUbo = UBO(device, backingBuffer = buffers["UBOBuffer"])
        with(matricesUbo) {
            name = "Default"
            members.put("ViewMatrix", { node.view })
            members.put("ModelMatrix", { node.world })
            members.put("ProjectionMatrix", { node.projection })
            members.put("MVP", { node.mvp })
            members.put("CamPosition", { scene.findObserver().position })
            members.put("isBillboard", { node.isBillboard.toInt() })

            requiredOffsetCount = 2
            createUniformBuffer(memoryProperties)
            sceneUBOs.put(node, this)
            s.UBOs.put("Default", this)
        }

        s = loadTexturesForNode(node, s)

        if (node.material != null) {
            val materialUbo = UBO(device, backingBuffer = buffers["UBOBuffer"])
            var materialType = 0

            if (node.material!!.textures.containsKey("ambient") && !s.defaultTexturesFor.contains("ambient")) {
                materialType = materialType or MATERIAL_HAS_AMBIENT
            }

            if (node.material!!.textures.containsKey("diffuse") && !s.defaultTexturesFor.contains("diffuse")) {
                materialType = materialType or MATERIAL_HAS_DIFFUSE
            }

            if (node.material!!.textures.containsKey("specular") && !s.defaultTexturesFor.contains("specular")) {
                materialType = materialType or MATERIAL_HAS_SPECULAR
            }

            if (node.material!!.textures.containsKey("normal") && !s.defaultTexturesFor.contains("normal")) {
                materialType = materialType or MATERIAL_HAS_NORMAL
            }

            with(materialUbo) {
                name = "BlinnPhongMaterial"
                members.put("Ka", { node.material!!.ambient })
                members.put("Kd", { node.material!!.diffuse })
                members.put("Ks", { node.material!!.specular })
                members.put("Shininess", { node.material!!.specularExponent })
                members.put("materialType", { materialType })

                requiredOffsetCount = 1
                createUniformBuffer(memoryProperties)
                s.UBOs.put("BlinnPhongMaterial", this)
            }
        } else {
            val materialUbo = UBO(device, backingBuffer = buffers["UBOBuffer"])
            val m = Material.DefaultMaterial()

            with(materialUbo) {
                name = "BlinnPhongMaterial"
                members.put("Ka", { m.ambient })
                members.put("Kd", { m.diffuse })
                members.put("Ks", { m.specular })
                members.put("Shininess", { m.specularExponent })
                members.put("materialType", { 0 })

                requiredOffsetCount = 1
                createUniformBuffer(memoryProperties)
                s.UBOs.put("BlinnPhongMaterial", this)
            }
        }

        s.initialized = true
        node.initialized = true
        node.metadata["VulkanRenderer"] = s

        node.material?.doubleSided?.let {
            if (it) {
                renderpasses.filter { it.value.passConfig.type == RenderConfigReader.RenderpassType.geometry }
                    .map { pass ->
                        val shaders = pass.value.passConfig.shaders
                        logger.info("initializing double-sided pipeline for ${node.name} from $shaders")

                        pass.value.initializePipeline("preferred-${node.name}",
                            shaders.map { VulkanShaderModule(device, "main", "shaders/" + it) },

                            settings = { pipeline ->
                                pipeline.rasterizationState.cullMode(VK_CULL_MODE_NONE)
                            },
                            vertexInputType = s.vertexDescription!!)

                    }
            }
        }

        if (s.vertexInputType == VertexDataKinds.coords_normals) {
            renderpasses.filter { it.value.passConfig.type == RenderConfigReader.RenderpassType.geometry }
                .map { pass ->
                    val shaders = pass.value.passConfig.shaders
                    logger.debug("initializing custom vertex input pipeline for ${node.name} from $shaders")

                    pass.value.initializePipeline("preferred-${node.name}",
                        shaders.map { VulkanShaderModule(device, "main", "shaders/" + it) },
                        vertexInputType = s.vertexDescription!!)
                }
        }

        val sp = node.metadata.values.find { it is ShaderPreference }
        if (sp != null) {
            renderpasses.filter { it.value.passConfig.type == RenderConfigReader.RenderpassType.geometry }
                .map { pass ->
                    val shaders = (sp as ShaderPreference).shaders
                    logger.info("initializing preferred pipeline for ${node.name} from $shaders")
                    pass.value.initializePipeline("preferred-${node.name}",
                        shaders.map { VulkanShaderModule(device, "main", "shaders/" + it + ".spv") },

                        vertexInputType = s.vertexDescription!!)
                }
        }

        return true
    }

    fun destroyNode(node: Node) {
        if(!node.metadata.containsKey("VulkanRenderer")) {
            return
        }

        val s = node.metadata["VulkanRenderer"] as VulkanObjectState

        s.UBOs.forEach { it.value.close() }

        if(node is HasGeometry) {
            s.vertexBuffers.forEach {
                it.value.close()
            }
        }
    }

    protected fun loadTexturesForNode(node: Node, s: VulkanObjectState): VulkanObjectState {
        val stats = hub?.get(SceneryElement.STATISTICS) as Statistics?

        if (node.lock.tryLock()) {
            node.material?.textures?.forEach {
                type, texture ->

                val slot = when (type) {
                    "ambient" -> 0
                    "diffuse" -> 1
                    "specular" -> 2
                    "normal" -> 3
                    "displacement" -> 4
                    else -> 0
                }

                val mipLevels = if(type == "ambient" || type == "diffuse" || type == "specular") {
                    3
                } else {
                    1
                }

                logger.debug("${node.name} will have $type texture from $texture in slot $slot")

                if (!textureCache.containsKey(texture) || node.material?.needsTextureReload!!) {
                    logger.trace("Loading texture $texture for ${node.name}")

                    val vkTexture = if (texture.startsWith("fromBuffer:")) {
                        val gt = node.material!!.transferTextures[texture.substringAfter("fromBuffer:")]

                        val t = VulkanTexture(device, physicalDevice, memoryProperties,
                            commandPools.Standard, queue,
                            gt!!.dimensions.x().toInt(), gt.dimensions.y().toInt(), 1,
                            mipLevels = mipLevels)
                        t.copyFrom(gt.contents)

                        t
                    } else {
                        val start = System.nanoTime()
                        val t = VulkanTexture.loadFromFile(device, physicalDevice, memoryProperties,
                            commandPools.Standard, queue, texture, true, mipLevels)
                        val duration = System.nanoTime() - start*1.0f
                        stats?.add("loadTexture", duration)

                        t
                    }

                    s.textures.put(type, vkTexture!!)
                    textureCache.put(texture, vkTexture)
                } else {
                    s.textures.put(type, textureCache[texture]!!)
                }
            }

            arrayOf("ambient", "diffuse", "specular", "normal", "displacement").forEach {
                if (!s.textures.containsKey(it)) {
                    s.textures.putIfAbsent(it, textureCache["DefaultTexture"])
                    s.defaultTexturesFor.add(it)
                }
            }

            s.texturesToDescriptorSet(device, descriptorSetLayouts["ObjectTextures"]!!,
                descriptorPool,
                targetBinding = 0)

            node.lock.unlock()
        }

        return s
    }

    protected fun prepareDefaultDescriptorSetLayouts(device: VkDevice): ConcurrentHashMap<String, Long> {
        val m = ConcurrentHashMap<String, Long>()

        m.put("default", VU.createDescriptorSetLayout(
            device,
            listOf(
                Pair(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1),
                Pair(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1),
                Pair(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1)),
            VK_SHADER_STAGE_ALL_GRAPHICS))

        m.put("LightParameters", VU.createDescriptorSetLayout(
            device,
            listOf(Pair(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1)),
            VK_SHADER_STAGE_ALL_GRAPHICS))

        m.put("ObjectTextures", VU.createDescriptorSetLayout(
            device,
            listOf(Pair(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 5)),
            VK_SHADER_STAGE_ALL_GRAPHICS))

        renderConfig.renderpasses.forEach { rp ->
            rp.value.inputs?.let {
                renderConfig.rendertargets?.let { rts ->
                    val rt = rts.get(it.first())!!

                    // create descriptor set layout that matches the render target
                    m.put("outputs-${it.first()}",
                        VU.createDescriptorSetLayout(device,
                            descriptorNum = rt.count(),
                            descriptorCount = 1,
                            type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                        ))
                }
            }
        }

        return m
    }

    protected fun prepareDescriptorSets(device: VkDevice, descriptorPool: Long) {
        this.descriptorSets.put("default",
            VU.createDescriptorSetDynamic(device, descriptorPool,
                descriptorSetLayouts["default"]!!, standardUBOs.count(),
                buffers["UBOBuffer"]!!))

        this.descriptorSets.put("LightParameters",
            VU.createDescriptorSetDynamic(device, descriptorPool,
                descriptorSetLayouts["LightParameters"]!!, 1,
                buffers["LightParametersBuffer"]!!))
    }

    protected fun prepareStandardVertexDescriptors(): ConcurrentHashMap<VertexDataKinds, VertexDescription> {
        val map = ConcurrentHashMap<VertexDataKinds, VertexDescription>()

        VertexDataKinds.values().forEach { kind ->
            var attributeDesc: VkVertexInputAttributeDescription.Buffer?
            var stride = 0

            when (kind) {
                VertexDataKinds.coords_none -> {
                    stride = 0
                    attributeDesc = null
                }

                VertexDataKinds.coords_normals -> {
                    stride = 3 + 3
                    attributeDesc = VkVertexInputAttributeDescription.calloc(2)

                    attributeDesc.get(1)
                        .binding(0)
                        .location(1)
                        .format(VK_FORMAT_R32G32B32_SFLOAT)
                        .offset(3 * 4)
                }

                VertexDataKinds.coords_normals_texcoords -> {
                    stride = 3 + 3 + 2
                    attributeDesc = VkVertexInputAttributeDescription.calloc(3)

                    attributeDesc.get(1)
                        .binding(0)
                        .location(1)
                        .format(VK_FORMAT_R32G32B32_SFLOAT)
                        .offset(3 * 4)

                    attributeDesc.get(2)
                        .binding(0)
                        .location(2)
                        .format(VK_FORMAT_R32G32_SFLOAT)
                        .offset(3 * 4 + 3 * 4)
                }

                VertexDataKinds.coords_texcoords -> {
                    stride = 3 + 2
                    attributeDesc = VkVertexInputAttributeDescription.calloc(2)

                    attributeDesc.get(1)
                        .binding(0)
                        .location(1)
                        .format(VK_FORMAT_R32G32_SFLOAT)
                        .offset(3 * 4)
                }
            }

            if (attributeDesc != null) {
                attributeDesc.get(0)
                    .binding(0)
                    .location(0)
                    .format(VK_FORMAT_R32G32B32_SFLOAT)
                    .offset(0)
            }

            val bindingDesc = if (attributeDesc != null) {
                VkVertexInputBindingDescription.calloc(1)
                    .binding(0)
                    .stride(stride * 4)
                    .inputRate(VK_VERTEX_INPUT_RATE_VERTEX)
            } else {
                null
            }

            val inputState = VkPipelineVertexInputStateCreateInfo.calloc()
                .sType(VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO)
                .pNext(NULL)
                .pVertexAttributeDescriptions(attributeDesc)
                .pVertexBindingDescriptions(bindingDesc)

            map.put(kind, VertexDescription(inputState, attributeDesc, bindingDesc))
        }

        return map
    }

    data class AttributeInfo(val format: Int, val elementByteSize: Int, val elementCount: Int)

    fun HashMap<String, () -> Any>.getFormatsAndRequiredAttributeSize(): List<AttributeInfo> {
        return this.map {
            val value = it.value.invoke()

            when (value.javaClass) {
                GLVector::class.java -> {
                    val v = value as GLVector
                    if (v.toFloatArray().size == 2) {
                        graphics.scenery.backends.vulkan.VulkanRenderer.AttributeInfo(VK_FORMAT_R32G32_SFLOAT, 4 * 2, 1)
                    } else if (v.toFloatArray().size == 4) {
                        graphics.scenery.backends.vulkan.VulkanRenderer.AttributeInfo(VK_FORMAT_R32G32B32A32_SFLOAT, 4 * 4, 1)
                    } else {
                        logger.error("Unsupported vector length for instancing: ${v.toFloatArray().size}")
                        graphics.scenery.backends.vulkan.VulkanRenderer.AttributeInfo(-1, -1, -1)
                    }
                }

                GLMatrix::class.java -> {
                    val m = value as GLMatrix
                    graphics.scenery.backends.vulkan.VulkanRenderer.AttributeInfo(VK_FORMAT_R32G32B32A32_SFLOAT, 4 * 4, m.floatArray.size / 4)
                }

                else -> {
                    logger.error("Unsupported type for instancing: ${value.javaClass.simpleName}")
                    graphics.scenery.backends.vulkan.VulkanRenderer.AttributeInfo(-1, -1, -1)
                }
            }
        }
    }

    protected fun vertexDescriptionFromInstancedNode(node: Node, template: VertexDescription): VertexDescription {
        logger.debug("Creating instanced vertex description for ${node.name}")

        val attributeDescs = template.attributeDescription
        val bindingDescs = template.bindingDescription

        val formatsAndAttributeSizes = node.instancedProperties.getFormatsAndRequiredAttributeSize()
        val newAttributesNeeded = formatsAndAttributeSizes.map { it.elementCount }.sum()

        val newAttributeDesc = VkVertexInputAttributeDescription
            .calloc(attributeDescs!!.capacity() + newAttributesNeeded)

        var position = 0
        var offset = 0

        (0..attributeDescs.capacity() - 1).forEachIndexed { i, attr ->
            newAttributeDesc[i].set(attributeDescs[i])
            offset += newAttributeDesc[i].offset()
            logger.debug("location(${newAttributeDesc[i].location()})")
            logger.debug("    .offset(${newAttributeDesc[i].offset()})")
            position = i
        }

        position = 3
        offset = 0

        formatsAndAttributeSizes.zip(node.instancedProperties.toList().reversed()).forEach {
            val attribInfo = it.first
            val property = it.second

            (0..attribInfo.elementCount - 1).forEach {
                newAttributeDesc[position]
                    .binding(1)
                    .location(position)
                    .format(attribInfo.format)
                    .offset(offset)

                logger.debug("location($position, ${it}/${attribInfo.elementCount}) for ${property.first}, type: ${property.second.invoke().javaClass.simpleName}")
                logger.debug("   .format(${attribInfo.format})")
                logger.debug("   .offset($offset)")

                offset += attribInfo.elementByteSize
                position++
            }
        }

        logger.debug("stride(${offset}), ${bindingDescs!!.capacity()}")

        val newBindingDesc = VkVertexInputBindingDescription.calloc(bindingDescs!!.capacity() + 1)
        newBindingDesc[0].set(bindingDescs[0])
        newBindingDesc[1]
            .binding(1)
            .stride(offset)
            .inputRate(VK_VERTEX_INPUT_RATE_INSTANCE)

        val inputState = VkPipelineVertexInputStateCreateInfo.calloc()
            .sType(VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO)
            .pNext(NULL)
            .pVertexAttributeDescriptions(newAttributeDesc)
            .pVertexBindingDescriptions(newBindingDesc)

        return VertexDescription(inputState, newAttributeDesc, newBindingDesc)
    }

    protected fun prepareDefaultTextures(device: VkDevice) {
        val t = VulkanTexture.loadFromFile(device, physicalDevice, memoryProperties, commandPools.Standard, queue,
            Renderer::class.java.getResource("DefaultTexture.png").path.toString(), true, 1)

        textureCache.put("DefaultTexture", t!!)
    }

    protected fun prepareRenderpassesFromConfig(config: RenderConfigReader.RenderConfig, width: Int, height: Int): LinkedHashMap<String, VulkanRenderpass> {
        // create all renderpasses first
        val passes = LinkedHashMap<String, VulkanRenderpass>()
        val framebuffers = ConcurrentHashMap<String, VulkanFramebuffer>()

        flow = renderConfig.createRenderpassFlow()

        config.createRenderpassFlow().map { passName ->
            val passConfig = config.renderpasses.get(passName)!!
            val pass = VulkanRenderpass(passName, config, device, descriptorPool, pipelineCache,
                memoryProperties, vertexDescriptors)

            // create framebuffer
            with(VU.newCommandBuffer(device, commandPools.Standard, autostart = true)) {
                config.rendertargets?.filter { it.key == passConfig.output }?.map { rt ->
                    logger.info("Creating render framebuffer ${rt.key} for pass ${passName}")

                    if(framebuffers.containsKey(rt.key)) {
                        logger.info("Reusing already created framebuffer")
                        pass.output.put(rt.key, framebuffers.get(rt.key)!!)
                    } else {

                        val framebuffer = VulkanFramebuffer(device, physicalDevice, commandPools.Standard,
                            width, height, this)

                        rt.value.forEach { att ->
                            logger.info(" + attachment ${att.key}, ${att.value.format.name}")

                            when (att.value.format) {
                                RenderConfigReader.TargetFormat.RGBA_Float32 -> framebuffer.addFloatRGBABuffer(att.key, 32)
                                RenderConfigReader.TargetFormat.RGBA_Float16 -> framebuffer.addFloatRGBABuffer(att.key, 16)
                                RenderConfigReader.TargetFormat.RGBA_Float11 -> framebuffer.addFloatRGBABuffer(att.key, 11)

                                RenderConfigReader.TargetFormat.RGB_Float32 -> framebuffer.addFloatRGBBuffer(att.key, 32)
                                RenderConfigReader.TargetFormat.RGB_Float16 -> framebuffer.addFloatRGBBuffer(att.key, 16)

                                RenderConfigReader.TargetFormat.RG_Float32 -> framebuffer.addFloatRGBuffer(att.key, 32)
                                RenderConfigReader.TargetFormat.RG_Float16 -> framebuffer.addFloatRGBuffer(att.key, 16)

                                RenderConfigReader.TargetFormat.RGBA_UInt16 -> framebuffer.addUnsignedByteRGBABuffer(att.key, 16)
                                RenderConfigReader.TargetFormat.RGBA_UInt8 -> framebuffer.addUnsignedByteRGBABuffer(att.key, 8)

                                RenderConfigReader.TargetFormat.Depth32 -> framebuffer.addDepthBuffer(att.key, 32)
                                RenderConfigReader.TargetFormat.Depth24 -> framebuffer.addDepthBuffer(att.key, 24)
                            }

                        }

                        framebuffer.createRenderpassAndFramebuffer()
                        framebuffer.outputDescriptorSet = VU.createRenderTargetDescriptorSet(device,
                            descriptorPool, descriptorSetLayouts["outputs-${rt.key}"]!!, rt.value, framebuffer)

                        pass.output.put(rt.key, framebuffer)
                        framebuffers.put(rt.key, framebuffer)
                    }
                }

                if (passConfig.output == "Viewport") {
                    // let's also create the default framebuffers
                    pass.commandBufferCount = swapchain!!.images!!.size

                    swapchain!!.images!!.forEachIndexed { i, image ->
                        val fb = VulkanFramebuffer(device, physicalDevice, commandPools.Standard,
                            width, height, this@with)

                        fb.addSwapchainAttachment("swapchain-$i", swapchain!!, i)
                        fb.addDepthBuffer("swapchain-$i-depth", 32)
                        fb.createRenderpassAndFramebuffer()

                        pass.output.put("Viewport-$i", fb)
                    }
                }

                pass.vulkanMetadata.clearValues.free()
                pass.vulkanMetadata.clearValues = VkClearValue.calloc(pass.output.values.first().attachments.count())
                pass.output.values.first().attachments.values.forEachIndexed { i, att ->
                    when (att.type) {
                        VulkanFramebuffer.VulkanFramebufferType.COLOR_ATTACHMENT -> {
                            pass.vulkanMetadata.clearValues[i].color().float32().put(pass.passConfig.clearColor.toFloatArray())
                        }
                        VulkanFramebuffer.VulkanFramebufferType.DEPTH_ATTACHMENT -> {
                            pass.vulkanMetadata.clearValues[i].depthStencil().set(pass.passConfig.depthClearValue, 0)
                        }
                    }
                }

                pass.vulkanMetadata.renderArea.extent().set(
                    (pass.passConfig.viewportSize.first*window.width).toInt(),
                    (pass.passConfig.viewportSize.second*window.height).toInt())
                pass.vulkanMetadata.renderArea.offset().set(
                    (pass.passConfig.viewportOffset.first*window.width).toInt(),
                    (pass.passConfig.viewportOffset.second*window.height).toInt())

                pass.vulkanMetadata.viewport[0].set(
                    (pass.passConfig.viewportOffset.first*window.width),
                    (pass.passConfig.viewportOffset.second*window.height),
                    (pass.passConfig.viewportSize.first*window.width),
                    (pass.passConfig.viewportSize.second*window.height),
                    0.0f, 1.0f)

                pass.vulkanMetadata.scissor[0].extent().set(
                    (pass.passConfig.viewportSize.first*window.width).toInt(),
                    (pass.passConfig.viewportSize.second*window.height).toInt())

                pass.vulkanMetadata.scissor[0].offset().set(
                    (pass.passConfig.viewportOffset.first*window.width).toInt(),
                    (pass.passConfig.viewportOffset.second*window.height).toInt())

                pass.semaphore = VU.run(memAllocLong(1), "vkCreateSemaphore") {
                     vkCreateSemaphore(device, semaphoreCreateInfo, null, this)
                }

                this.endCommandBuffer(device, commandPools.Standard, this@VulkanRenderer.queue, flush = true)
            }

            passes.put(passName, pass)
        }

        // connect inputs with each other
        passes.forEach { pass ->
            val passConfig = config.renderpasses.get(pass.key)!!

            passConfig.inputs?.forEach { inputTarget ->
                passes.filter {
                    it.value.output.keys.contains(inputTarget)
                }.forEach { pass.value.inputs.put(inputTarget, it.value.output.get(inputTarget)!!) }
            }

            with(pass.value) {
                initializeInputAttachmentDescriptorSetLayouts()
                initializeShaderParameterDescriptorSetLayouts(settings)

                initializeDefaultPipeline()
            }
        }

        return passes
    }

    protected fun prepareStandardSemaphores(device: VkDevice): ConcurrentHashMap<StandardSemaphores, Array<Long>> {
        val map = ConcurrentHashMap<StandardSemaphores, Array<Long>>()

        StandardSemaphores.values().forEach {
            map.put(it, swapchain!!.images!!.map {
                VU.run(memAllocLong(1), "Semaphore for $it") {
                    vkCreateSemaphore(device, semaphoreCreateInfo, null, this)
                }
            }.toTypedArray())
        }

        return map
    }

    fun Long.hex(): String {
        return String.format("0x%X", this)
    }

    private fun pollEvents() {
        if (glfwWindowShouldClose(window.glfwWindow!!)) {
            this.shouldClose = true
        }

        glfwPollEvents()

        if (swapchainRecreator.mustRecreate) {
            swapchainRecreator.recreate()
            frames = 0
        }
    }

    fun beginFrame() {
        // this will wait infinite time or until an error occurs, then signal
        // that an image is available
        var err = vkAcquireNextImageKHR(device, swapchain!!.handle, UINT64_MAX,
            semaphores[StandardSemaphores.present_complete]!!.get(0),
            VK_NULL_HANDLE, swapchainImage)

        if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR) {
            swapchainRecreator.mustRecreate = true
        } else if (err != VK_SUCCESS) {
            throw AssertionError("Failed to acquire next swapchain image: " + VU.translate(err))
        }
    }


    fun submitFrame(queue: VkQueue, pass: VulkanRenderpass, commandBuffer: VulkanCommandBuffer, present: PresentHelpers) {
        val submitInfo = VkSubmitInfo.calloc()
            .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
            .pNext(NULL)
            .waitSemaphoreCount(1)
            .pWaitSemaphores(present.waitSemaphore)
            .pWaitDstStageMask(present.waitStages)
            .pCommandBuffers(present.commandBuffers)
            .pSignalSemaphores(present.signalSemaphore)

        // Submit to the graphics queue
        var err = vkQueueSubmit(queue, submitInfo, commandBuffer.fence.get(0))
        if (err != VK_SUCCESS) {
            throw AssertionError("Frame $frames: Failed to submit render queue: " + VU.translate(err))
        }

        commandBuffer.submitted = true

        // Present the current buffer to the swap chain
        // This will display the image
        pSwapchains.put(0, swapchain!!.handle)

        // Info struct to present the current swapchain image to the display
        val presentInfo = VkPresentInfoKHR.calloc()
            .sType(VK_STRUCTURE_TYPE_PRESENT_INFO_KHR)
            .pNext(NULL)
            .pWaitSemaphores(present.signalSemaphore)
            .swapchainCount(pSwapchains.remaining())
            .pSwapchains(pSwapchains)
            .pImageIndices(swapchainImage)
            .pResults(null)

        err = vkQueuePresentKHR(queue, presentInfo)
        if (err != VK_SUCCESS) {
            throw AssertionError("Failed to present the swapchain image: " + VU.translate(err))
        }

        if (screenshotRequested) {
            // default image format is 32bit BGRA
            val imageByteSize = window.width * window.height * 4L
            val screenshotBuffer = VU.createBuffer(device,
                memoryProperties, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                wantAligned = true,
                allocationSize = imageByteSize)

            with(VU.newCommandBuffer(device, commandPools.Render, autostart = true)) {
                val subresource = VkImageSubresourceLayers.calloc()
                    .aspectMask(VK_IMAGE_ASPECT_COLOR_BIT)
                    .mipLevel(0)
                    .baseArrayLayer(0)
                    .layerCount(1)

                val regions = VkBufferImageCopy.calloc(1)
                    .bufferRowLength(0)
                    .bufferImageHeight(0)
                    .imageOffset(VkOffset3D.calloc().set(0, 0, 0))
                    .imageExtent(VkExtent3D.calloc().set(window.width, window.height, 1))
                    .imageSubresource(subresource)

                val image = swapchain!!.images!![pass.getReadPosition()]

                VulkanTexture.transitionLayout(image,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    commandBuffer = this)

                vkCmdCopyImageToBuffer(this, image,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    screenshotBuffer.buffer,
                    regions)

                VulkanTexture.transitionLayout(image,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    commandBuffer = this)

                this.endCommandBuffer(device, commandPools.Render, queue,
                    flush = true, dealloc = true)
            }

            vkDeviceWaitIdle(device)

            val imageBuffer = memAlloc(imageByteSize.toInt())
            screenshotBuffer.copyTo(imageBuffer)
            screenshotBuffer.close()

            thread {
                try {
                    val file = File(System.getProperty("user.home"), "Desktop" + File.separator + "$applicationName - ${SimpleDateFormat("yyyy-MM-dd HH.mm.ss").format(Date())}.png")
                    imageBuffer.rewind()

                    val imageArray = ByteArray(imageBuffer.remaining())
                    imageBuffer.get(imageArray)
                    val shifted = ByteArray(imageArray.size)

                    // swizzle BGRA -> ABGR
                    for (i in 0..shifted.size - 1 step 4) {
                        shifted[i] = imageArray[i + 3]
                        shifted[i + 1] = imageArray[i]
                        shifted[i + 2] = imageArray[i + 1]
                        shifted[i + 3] = imageArray[i + 2]
                    }

                    val image = BufferedImage(window.width, window.height, BufferedImage.TYPE_4BYTE_ABGR)
                    val imgData = (image.raster.dataBuffer as DataBufferByte).data
                    System.arraycopy(shifted, 0, imgData, 0, shifted.size)

                    ImageIO.write(image, "png", file)
                    logger.info("Screenshot saved to ${file.absolutePath}")
                } catch (e: Exception) {
                    System.err.println("Unable to take screenshot: ")
                    e.printStackTrace()
                } finally {
                    memFree(imageBuffer)
                }
            }

            screenshotRequested = false
        }

        pass.nextSwapchainImage()

        submitInfo.free()
        presentInfo.free()
    }

    /**
     * This function renders the scene
     *
     * @param[scene] The scene to render.
     */
    override fun render() {
        pollEvents()

        // check whether scene is already initialized
        if (scene.children.count() == 0 || scene.initialized == false) {
            initializeScene()

            Thread.sleep(200)
            return
        }

        if(toggleFullscreen) {
            vkDeviceWaitIdle(device)

            switchFullscreen()
            toggleFullscreen = false
            return
        }

        if(shouldClose) {
            // stop all
            vkDeviceWaitIdle(device)
            return
        }

        updateDefaultUBOs(device)
        updateInstanceBuffers()

        beginFrame()

        // firstWaitSemaphore is now the render_complete semaphore of the previous pass
        firstWaitSemaphore.put(0, semaphores[StandardSemaphores.present_complete]!!.get(0))

        flow.take(flow.size - 1).forEachIndexed { i, t ->
            logger.debug("Running pass {}", t)
            val target = renderpasses[t]!!
            val commandBuffer = target.commandBuffer

            if (commandBuffer.submitted) {
                commandBuffer.waitForFence()
            }

            commandBuffer.resetFence()

            when (target.passConfig.type) {
                RenderConfigReader.RenderpassType.geometry -> recordSceneRenderCommands(device, target, commandBuffer)
                RenderConfigReader.RenderpassType.quad -> recordPostprocessRenderCommands(device, target, commandBuffer)
            }

            target.updateShaderParameters()

            ph.commandBuffers.put(0, commandBuffer.commandBuffer)
            ph.signalSemaphore.put(0, target.semaphore)
            ph.waitStages.put(0, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)
            ph.waitStages.put(1, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)

            val si = VkSubmitInfo.calloc()
                .sType(VK_STRUCTURE_TYPE_SUBMIT_INFO)
                .pNext(NULL)
                .waitSemaphoreCount(1)
                .pWaitDstStageMask(ph.waitStages)
                .pCommandBuffers(ph.commandBuffers)
                .pSignalSemaphores(ph.signalSemaphore)
                .pWaitSemaphores(firstWaitSemaphore)

            vkQueueSubmit(queue, si, commandBuffer.fence.get(0))

            commandBuffer.submitted = true
            firstWaitSemaphore.put(0, target.semaphore)

            si.free()
        }

        val viewportPass = renderpasses.values.last()
        val viewportCommandBuffer = viewportPass.commandBuffer

        if (viewportCommandBuffer.submitted) {
            viewportCommandBuffer.waitForFence()
        }

        viewportCommandBuffer.resetFence()

        when (viewportPass.passConfig.type) {
            RenderConfigReader.RenderpassType.geometry -> recordSceneRenderCommands(device, viewportPass, viewportCommandBuffer)
            RenderConfigReader.RenderpassType.quad -> recordPostprocessRenderCommands(device, viewportPass, viewportCommandBuffer)
        }

        viewportPass.updateShaderParameters()

        ph.commandBuffers.put(0, viewportCommandBuffer.commandBuffer)
        ph.waitStages.put(0, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT)
        ph.signalSemaphore.put(0, semaphores[StandardSemaphores.render_complete]!!.get(0))
        ph.waitSemaphore.put(0, firstWaitSemaphore.get(0))

        submitFrame(queue, viewportPass, viewportCommandBuffer, ph)

        updateTimings()
    }

    private fun updateTimings() {
        val thisTime = System.nanoTime()
        time += (thisTime - lastTime) / 1E9f
        lastTime = thisTime

        frames++
        totalFrames++
    }

    private fun createInstance(requiredExtensions: PointerBuffer): VkInstance {
        val appInfo = VkApplicationInfo.calloc()
            .sType(VK_STRUCTURE_TYPE_APPLICATION_INFO)
            .pApplicationName(memUTF8(applicationName))
            .pEngineName(memUTF8("scenery"))
            .apiVersion(VK_MAKE_VERSION(1, 0, 24))

        val ppEnabledExtensionNames = memAllocPointer(requiredExtensions.remaining() + 1)
        ppEnabledExtensionNames.put(requiredExtensions)
        val VK_EXT_DEBUG_REPORT_EXTENSION = memUTF8(VK_EXT_DEBUG_REPORT_EXTENSION_NAME)
        ppEnabledExtensionNames.put(VK_EXT_DEBUG_REPORT_EXTENSION)
        ppEnabledExtensionNames.flip()
        val ppEnabledLayerNames = memAllocPointer(layers.size)
        var i = 0
        while (validation && i < layers.size) {
            ppEnabledLayerNames.put(layers[i])
            i++
        }
        ppEnabledLayerNames.flip()

        val pCreateInfo = VkInstanceCreateInfo.calloc()
            .sType(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO)
            .pNext(NULL)
            .pApplicationInfo(appInfo)
            .ppEnabledExtensionNames(ppEnabledExtensionNames)
            .ppEnabledLayerNames(ppEnabledLayerNames)

        val pInstance = memAllocPointer(1)
        val err = vkCreateInstance(pCreateInfo, null, pInstance)
        val instance = pInstance.get(0)
        memFree(pInstance)
        if (err != VK_SUCCESS) {
            throw AssertionError("Failed to create VkInstance: " + VU.translate(err))
        }
        val ret = VkInstance(instance, pCreateInfo)
        pCreateInfo.free()
        memFree(ppEnabledLayerNames)
        memFree(VK_EXT_DEBUG_REPORT_EXTENSION)
        memFree(ppEnabledExtensionNames)
        memFree(appInfo.pApplicationName())
        memFree(appInfo.pEngineName())
        appInfo.free()
        return ret
    }

    private fun setupDebugging(instance: VkInstance, flags: Int, callback: VkDebugReportCallbackEXT): Long {
        val dbgCreateInfo = VkDebugReportCallbackCreateInfoEXT.calloc()
            .sType(VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT)
            .pNext(NULL)
            .pfnCallback(callback)
            .pUserData(NULL)
            .flags(flags)

        val pCallback = memAllocLong(1)
        val err = vkCreateDebugReportCallbackEXT(instance, dbgCreateInfo, null, pCallback)
        val callbackHandle = pCallback.get(0)
        memFree(pCallback)
        dbgCreateInfo.free()
        if (err != VK_SUCCESS) {
            throw AssertionError("Failed to create VkInstance: " + VU.translate(err))
        }
        return callbackHandle
    }


    private fun getPhysicalDevice(instance: VkInstance): VkPhysicalDevice {
        val pPhysicalDeviceCount = memAllocInt(1)
        var err = vkEnumeratePhysicalDevices(instance, pPhysicalDeviceCount, null)

        if (err != VK_SUCCESS) {
            throw AssertionError("Failed to get number of physical devices: " + VU.translate(err))
        }

        if (pPhysicalDeviceCount.get(0) < 1) {
            throw AssertionError("No Vulkan-compatible devices found!")
        }

        val pPhysicalDevices = memAllocPointer(pPhysicalDeviceCount.get(0))
        err = vkEnumeratePhysicalDevices(instance, pPhysicalDeviceCount, pPhysicalDevices)

        val devicePreference = System.getProperty("scenery.VulkanRenderer.Device", "0").toInt()

        logger.info("Physical devices: ")
        val properties: VkPhysicalDeviceProperties = VkPhysicalDeviceProperties.calloc()

        for (i in 0..pPhysicalDeviceCount.get(0) - 1) {
            val device = VkPhysicalDevice(pPhysicalDevices.get(i), instance)

            vkGetPhysicalDeviceProperties(device, properties)
            logger.info("  $i: ${VU.vendorToString(properties.vendorID())} ${properties.deviceNameString()} (${VU.deviceTypeToString(properties.deviceType())}, driver version ${VU.driverVersionToString(properties.driverVersion())}, Vulkan API ${VU.driverVersionToString(properties.apiVersion())}) ${if (devicePreference == i) {
                "(selected)"
            } else {
                ""
            }}")
        }

        val physicalDevice = pPhysicalDevices.get(devicePreference)

        memFree(pPhysicalDeviceCount)
        memFree(pPhysicalDevices)
        properties.free()

        if (err != VK_SUCCESS) {
            throw AssertionError("Failed to get physical devices: " + VU.translate(err))
        }

        return VkPhysicalDevice(physicalDevice, instance)
    }

    private fun createDeviceAndGetGraphicsQueueFamily(physicalDevice: VkPhysicalDevice): DeviceAndGraphicsQueueFamily {
        val pQueueFamilyPropertyCount = memAllocInt(1)
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, pQueueFamilyPropertyCount, null)
        val queueCount = pQueueFamilyPropertyCount.get(0)
        val queueProps = VkQueueFamilyProperties.calloc(queueCount)
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, pQueueFamilyPropertyCount, queueProps)
        memFree(pQueueFamilyPropertyCount)

        var graphicsQueueFamilyIndex: Int
        graphicsQueueFamilyIndex = 0
        while (graphicsQueueFamilyIndex < queueCount) {
            if (queueProps.get(graphicsQueueFamilyIndex).queueFlags() and VK_QUEUE_GRAPHICS_BIT !== 0)
                break
            graphicsQueueFamilyIndex++
        }
        queueProps.free()

        val pQueuePriorities = memAllocFloat(1).put(0.0f)
        pQueuePriorities.flip()
        val queueCreateInfo = VkDeviceQueueCreateInfo.calloc(1)
            .sType(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO)
            .queueFamilyIndex(graphicsQueueFamilyIndex)
            .pQueuePriorities(pQueuePriorities)

        val extensions = memAllocPointer(1)
        val VK_KHR_SWAPCHAIN_EXTENSION = memUTF8(VK_KHR_SWAPCHAIN_EXTENSION_NAME)
        extensions.put(VK_KHR_SWAPCHAIN_EXTENSION)
        extensions.flip()
        val ppEnabledLayerNames = memAllocPointer(layers.size)
        var i = 0
        while (validation && i < layers.size) {
            ppEnabledLayerNames.put(layers[i])
            i++
        }
        ppEnabledLayerNames.flip()

        if (validation) {
            logger.warn("Enabled Vulkan API validations. Expect degraded performance.")
        }

        val deviceCreateInfo = VkDeviceCreateInfo.calloc()
            .sType(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO)
            .pNext(NULL)
            .pQueueCreateInfos(queueCreateInfo)
            .ppEnabledExtensionNames(extensions)
            .ppEnabledLayerNames(ppEnabledLayerNames)

        val pDevice = memAllocPointer(1)
        val err = vkCreateDevice(physicalDevice, deviceCreateInfo, null, pDevice)
        val device = pDevice.get(0)
        memFree(pDevice)

        if (err != VK_SUCCESS) {
            throw AssertionError("Failed to create device: " + VU.translate(err))
        }

        val memoryProperties = VkPhysicalDeviceMemoryProperties.calloc()
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, memoryProperties)

        val ret = DeviceAndGraphicsQueueFamily()
        ret.device = VkDevice(device, physicalDevice, deviceCreateInfo)
        ret.queueFamilyIndex = graphicsQueueFamilyIndex
        ret.memoryProperties = memoryProperties

        deviceCreateInfo.free()
        memFree(ppEnabledLayerNames)
        memFree(VK_KHR_SWAPCHAIN_EXTENSION)
        memFree(extensions)
        memFree(pQueuePriorities)
        queueCreateInfo.free()

        return ret
    }


    private fun getColorFormatAndSpace(physicalDevice: VkPhysicalDevice, surface: Long): ColorFormatAndSpace {
        val pQueueFamilyPropertyCount = memAllocInt(1)
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, pQueueFamilyPropertyCount, null)
        val queueCount = pQueueFamilyPropertyCount.get(0)
        val queueProps = VkQueueFamilyProperties.calloc(queueCount)
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, pQueueFamilyPropertyCount, queueProps)
        memFree(pQueueFamilyPropertyCount)

        // Iterate over each queue to learn whether it supports presenting:
        val supportsPresent = memAllocInt(queueCount)
        for (i in 0..queueCount - 1) {
            supportsPresent.position(i)
            val err = vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, supportsPresent)
            if (err != VK_SUCCESS) {
                throw AssertionError("Failed to physical device surface support: " + VU.translate(err))
            }
        }

        // Search for a graphics and a present queue in the array of queue families, try to find one that supports both
        var graphicsQueueNodeIndex = Integer.MAX_VALUE
        var presentQueueNodeIndex = Integer.MAX_VALUE
        for (i in 0..queueCount - 1) {
            if (queueProps.get(i).queueFlags() and VK_QUEUE_GRAPHICS_BIT !== 0) {
                if (graphicsQueueNodeIndex == Integer.MAX_VALUE) {
                    graphicsQueueNodeIndex = i
                }
                if (supportsPresent.get(i) === VK_TRUE) {
                    graphicsQueueNodeIndex = i
                    presentQueueNodeIndex = i
                    break
                }
            }
        }
        queueProps.free()
        if (presentQueueNodeIndex == Integer.MAX_VALUE) {
            // If there's no queue that supports both present and graphics try to find a separate present queue
            for (i in 0..queueCount - 1) {
                if (supportsPresent.get(i) === VK_TRUE) {
                    presentQueueNodeIndex = i
                    break
                }
            }
        }
        memFree(supportsPresent)

        // Generate error if could not find both a graphics and a present queue
        if (graphicsQueueNodeIndex == Integer.MAX_VALUE) {
            throw AssertionError("No graphics queue found")
        }
        if (presentQueueNodeIndex == Integer.MAX_VALUE) {
            throw AssertionError("No presentation queue found")
        }
        if (graphicsQueueNodeIndex != presentQueueNodeIndex) {
            throw AssertionError("Presentation queue != graphics queue")
        }

        // Get list of supported formats
        val pFormatCount = memAllocInt(1)
        var err = vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, pFormatCount, null)
        val formatCount = pFormatCount.get(0)
        if (err != VK_SUCCESS) {
            throw AssertionError("Failed to query number of physical device surface formats: " + VU.translate(err))
        }

        val surfFormats = VkSurfaceFormatKHR.calloc(formatCount)
        err = vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, pFormatCount, surfFormats)
        memFree(pFormatCount)
        if (err != VK_SUCCESS) {
            throw AssertionError("Failed to query physical device surface formats: " + VU.translate(err))
        }

        val colorFormat: Int
        if (formatCount == 1 && surfFormats.get(0).format() === VK_FORMAT_UNDEFINED) {
//            colorFormat = VK_FORMAT_B8G8R8A8_UNORM
            colorFormat = VK_FORMAT_B8G8R8A8_SRGB
        } else {
//            colorFormat = surfFormats.get(0).format()
            colorFormat = VK_FORMAT_B8G8R8A8_SRGB
        }
        val colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR//surfFormats.get(0).colorSpace()
        surfFormats.free()

        val ret = ColorFormatAndSpace()
        ret.colorFormat = colorFormat
        ret.colorSpace = colorSpace
        return ret
    }

    private fun createCommandPool(device: VkDevice, queueNodeIndex: Int): Long {
        val cmdPoolInfo = VkCommandPoolCreateInfo.calloc()
            .sType(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO)
            .queueFamilyIndex(queueNodeIndex)
            .flags(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)

        val pCmdPool = memAllocLong(1)
        val err = vkCreateCommandPool(device, cmdPoolInfo, null, pCmdPool)
        val commandPool = pCmdPool.get(0)
        cmdPoolInfo.free()
        memFree(pCmdPool)
        if (err != VK_SUCCESS) {
            throw AssertionError("Failed to create command pool: " + VU.translate(err))
        }
        return commandPool
    }

    private fun destroyCommandPool(device: VkDevice, commandPool: Long) {
        vkDestroyCommandPool(device, commandPool, null)
    }

    private fun createSwapChain(device: VkDevice, physicalDevice: VkPhysicalDevice, surface: Long, oldSwapChain: Long, newWidth: Int,
                                newHeight: Int, colorFormat: Int, colorSpace: Int): Swapchain {
        var err: Int
        // Get physical device surface properties and formats
        val surfCaps = VkSurfaceCapabilitiesKHR.calloc()
        err = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, surfCaps)
        if (err != VK_SUCCESS) {
            throw AssertionError("Failed to get physical device surface capabilities: " + VU.translate(err))
        }

        val pPresentModeCount = memAllocInt(1)
        err = vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, pPresentModeCount, null)
        val presentModeCount = pPresentModeCount.get(0)
        if (err != VK_SUCCESS) {
            throw AssertionError("Failed to get number of physical device surface presentation modes: " + VU.translate(err))
        }

        val pPresentModes = memAllocInt(presentModeCount)
        err = vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, pPresentModeCount, pPresentModes)
        memFree(pPresentModeCount)
        if (err != VK_SUCCESS) {
            throw AssertionError("Failed to get physical device surface presentation modes: " + VU.translate(err))
        }

        // Try to use mailbox mode. Low latency and non-tearing
        var swapchainPresentMode = VK_PRESENT_MODE_FIFO_KHR
        for (i in 0..presentModeCount - 1) {
            if (pPresentModes.get(i) === VK_PRESENT_MODE_MAILBOX_KHR) {
                swapchainPresentMode = VK_PRESENT_MODE_MAILBOX_KHR
                break
            }
            if (swapchainPresentMode != VK_PRESENT_MODE_MAILBOX_KHR && pPresentModes.get(i) === VK_PRESENT_MODE_IMMEDIATE_KHR) {
                swapchainPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR
            }
        }
        memFree(pPresentModes)

        // Determine the number of images
        var desiredNumberOfSwapchainImages = surfCaps.minImageCount() + 1
        if (surfCaps.maxImageCount() > 0 && desiredNumberOfSwapchainImages > surfCaps.maxImageCount()) {
            desiredNumberOfSwapchainImages = surfCaps.maxImageCount()
        }

        val currentExtent = surfCaps.currentExtent()
        val currentWidth = currentExtent.width()
        val currentHeight = currentExtent.height()
        if (currentWidth != -1 && currentHeight != -1) {
            window.width = currentWidth
            window.height = currentHeight
        } else {
            window.width = newWidth
            window.height = newHeight
        }

        val preTransform: Int
        if (surfCaps.supportedTransforms() and VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR !== 0) {
            preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
        } else {
            preTransform = surfCaps.currentTransform()
        }
        surfCaps.free()

        val swapchainCI = VkSwapchainCreateInfoKHR.calloc()
            .sType(VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR)
            .pNext(NULL)
            .surface(surface)
            .minImageCount(desiredNumberOfSwapchainImages)
            .imageFormat(colorFormat)
            .imageColorSpace(colorSpace)
            .imageUsage(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT or VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
            .preTransform(preTransform)
            .imageArrayLayers(1)
            .imageSharingMode(VK_SHARING_MODE_EXCLUSIVE)
            .pQueueFamilyIndices(null)
            .presentMode(swapchainPresentMode)
            .oldSwapchain(oldSwapChain)
            .clipped(true)
            .compositeAlpha(VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)

        swapchainCI.imageExtent().width(window.width).height(window.height)
        val pSwapChain = memAllocLong(1)
        err = vkCreateSwapchainKHR(device, swapchainCI, null, pSwapChain)
        swapchainCI.free()
        val swapChain = pSwapChain.get(0)
        memFree(pSwapChain)
        if (err != VK_SUCCESS) {
            throw AssertionError("Failed to create swap chain: " + VU.translate(err))
        }

        // If we just re-created an existing swapchain, we should destroy the old swapchain at this point.
        // Note: destroying the swapchain also cleans up all its associated presentable images once the platform is done with them.
        if (oldSwapChain != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(device, oldSwapChain, null)
        }

        val pImageCount = memAllocInt(1)
        err = vkGetSwapchainImagesKHR(device, swapChain, pImageCount, null)
        val imageCount = pImageCount.get(0)
        if (err != VK_SUCCESS) {
            throw AssertionError("Failed to get number of swapchain images: " + VU.translate(err))
        }

        val pSwapchainImages = memAllocLong(imageCount)
        err = vkGetSwapchainImagesKHR(device, swapChain, pImageCount, pSwapchainImages)
        if (err != VK_SUCCESS) {
            throw AssertionError("Failed to get swapchain images: " + VU.translate(err))
        }
        memFree(pImageCount)

        val images = LongArray(imageCount)
        val imageViews = LongArray(imageCount)
        val colorAttachmentView = VkImageViewCreateInfo.calloc()
            .sType(VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO)
            .pNext(NULL)
            .format(colorFormat)
            .viewType(VK_IMAGE_VIEW_TYPE_2D)
            .flags(VK_FLAGS_NONE)

        colorAttachmentView.components()
            .r(VK_COMPONENT_SWIZZLE_R)
            .g(VK_COMPONENT_SWIZZLE_G)
            .b(VK_COMPONENT_SWIZZLE_B)
            .a(VK_COMPONENT_SWIZZLE_A)

        colorAttachmentView.subresourceRange()
            .aspectMask(VK_IMAGE_ASPECT_COLOR_BIT)
            .baseMipLevel(0)
            .levelCount(1)
            .baseArrayLayer(0)
            .layerCount(1)

        with(VU.newCommandBuffer(device, commandPools.Standard, autostart = true)) {
            for (i in 0..imageCount - 1) {
                images[i] = pSwapchainImages.get(i)

                VU.setImageLayout(this, images[i],
                    aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    oldImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    newImageLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
                colorAttachmentView.image(images[i])

                imageViews[i] = VU.run(memAllocLong(1), "create image view",
                    { vkCreateImageView(device, colorAttachmentView, null, this) })
            }

            this.endCommandBuffer(device, commandPools.Standard, queue,
                flush = true, dealloc = true)
        }

        colorAttachmentView.free()
        memFree(pSwapchainImages)

        val ret = Swapchain()
        ret.images = images
        ret.imageViews = imageViews
        ret.handle = swapChain
        return ret
    }

    private fun createVertexBuffers(device: VkDevice, node: Node, state: VulkanObjectState): VulkanObjectState {
        val n = node as HasGeometry

        if (n.texcoords.remaining() == 0) {
            n.texcoords = memAlloc(4 * n.vertices.remaining() / n.vertexSize * n.texcoordSize).asFloatBuffer()
        }

        val vertexAllocationBytes = 4 * (n.vertices.remaining() + n.normals.remaining() + n.texcoords.remaining())
        val indexAllocationBytes = 4 * n.indices.remaining()
        val fullAllocationBytes = vertexAllocationBytes + indexAllocationBytes

        val stridedBuffer = memAlloc(fullAllocationBytes)

        val fb = stridedBuffer.asFloatBuffer()
        val ib = stridedBuffer.asIntBuffer()

        state.vertexCount = n.vertices.remaining() / n.vertexSize
        logger.trace("${node.name} has ${n.vertices.remaining()} floats and ${n.texcoords.remaining() / n.texcoordSize} remaining")

        for (index in 0..n.vertices.remaining() - 1 step 3) {
            fb.put(n.vertices.get())
            fb.put(n.vertices.get())
            fb.put(n.vertices.get())

            fb.put(n.normals.get())
            fb.put(n.normals.get())
            fb.put(n.normals.get())

            if (n.texcoords.remaining() > 0) {
                fb.put(n.texcoords.get())
                fb.put(n.texcoords.get())
            }
        }

        logger.trace("Adding ${n.indices.remaining() * 4} bytes to strided buffer")
        if (n.indices.remaining() > 0) {
            state.isIndexed = true
            ib.position(vertexAllocationBytes / 4)

            for (index in 0..n.indices.remaining() - 1) {
                ib.put(n.indices.get())
            }
        }

        logger.trace("Strided buffer is now at ${stridedBuffer.remaining()} bytes")

        n.vertices.flip()
        n.normals.flip()
        n.texcoords.flip()
        n.indices.flip()

        val stagingBuffer = VU.createBuffer(device,
            this.memoryProperties,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            wantAligned = false,
            allocationSize = fullAllocationBytes * 1L)

        stagingBuffer.copyFrom(stridedBuffer)

        val vertexBuffer = VU.createBuffer(device,
            this.memoryProperties,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT or VK_BUFFER_USAGE_INDEX_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            wantAligned = false,
            allocationSize = fullAllocationBytes * 1L)

        with(VU.newCommandBuffer(device, commandPools.Standard, autostart = true)) {
            val copyRegion = VkBufferCopy.calloc(1)
                .size(fullAllocationBytes * 1L)

            vkCmdCopyBuffer(this,
                stagingBuffer.buffer,
                vertexBuffer.buffer,
                copyRegion)

            copyRegion.free()
            this.endCommandBuffer(device, commandPools.Standard, queue, flush = true, dealloc = true)
        }

        state.vertexBuffers.put("vertex+index", vertexBuffer)
        state.indexOffset = vertexAllocationBytes
        state.indexCount = n.indices.remaining()

        memFree(stridedBuffer)
        stagingBuffer.close()

        return state
    }

    private fun createInstanceBuffer(device: VkDevice, parentNode: Node, state: VulkanObjectState): VulkanObjectState {
        val instances = ArrayList<Node>()
        val cam = scene.findObserver()

        scene.discover(scene, { n -> n.instanceOf == parentNode }).forEach {
            instances.add(it)
        }

        if (instances.size < 1) {
            logger.info("$parentNode has no child instances attached, returning.")
            return state
        }

        // first we create a fake UBO to gauge the size of the needed properties
        val ubo = UBO(device)
        ubo.fromInstance(instances.first())

        val instanceBufferSize = ubo.getSize() * instances.size

        logger.debug("$parentNode has ${instances.size} child instances with ${ubo.getSize()} bytes each.")
        logger.debug("Creating staging buffer...")

        val stagingBuffer = VU.createBuffer(device,
            this.memoryProperties,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            wantAligned = false,
            allocationSize = instanceBufferSize * 1L)

        instances.forEach { node ->
            node.updateWorld(true, false)

            node.projection.copyFrom(cam.projection)
            node.projection.set(1, 1, -1.0f * cam.projection.get(1, 1))

            node.modelView.copyFrom(cam.view)
            node.modelView.mult(node.world)

            node.mvp.copyFrom(node.projection)
            node.mvp.mult(node.modelView)

            val instanceUbo = UBO(device, backingBuffer = stagingBuffer)
            instanceUbo.fromInstance(node)
            instanceUbo.createUniformBuffer(memoryProperties)
            instanceUbo.populate()
        }

        logger.debug("Copying from staging buffer")
        stagingBuffer.copyFromStagingBuffer()

        // the actual instance buffer is kept device-local for performance reasons
        val instanceBuffer = VU.createBuffer(device,
            this.memoryProperties,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            wantAligned = false,
            allocationSize = instanceBufferSize * 1L)

        with(VU.newCommandBuffer(device, commandPools.Standard, autostart = true)) {
            val copyRegion = VkBufferCopy.calloc(1)
                .size(instanceBufferSize * 1L)

            vkCmdCopyBuffer(this,
                stagingBuffer.buffer,
                instanceBuffer.buffer,
                copyRegion)

            this.endCommandBuffer(device, commandPools.Standard, queue, flush = true, dealloc = true)
        }

        state.vertexBuffers.put("instance", instanceBuffer)
        state.instanceCount = instances.size

        stagingBuffer.close()

        logger.debug("Instance buffer creation done")

        return state
    }

    private fun updateInstanceBuffer(device: VkDevice, parentNode: Node, state: VulkanObjectState): VulkanObjectState {
        val instances = ArrayList<Node>()
        val cam = scene.findObserver()

        scene.discover(scene, { n -> n.instanceOf == parentNode }).forEach {
            instances.add(it)
        }

        if (instances.size < 1) {
            logger.debug("$parentNode has no child instances attached, returning.")
            return state
        }

        // first we create a fake UBO to gauge the size of the needed properties
        val ubo = UBO(device)
        ubo.fromInstance(instances.first())

        val instanceBufferSize = ubo.getSize() * instances.size

        val stagingBuffer = VU.createBuffer(device,
            this.memoryProperties,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            wantAligned = true,
            allocationSize = instanceBufferSize * 1L)

        instances.forEach { node ->
            node.updateWorld(true, false)

            node.projection.copyFrom(cam.projection)
            node.projection.set(1, 1, -1.0f * cam.projection.get(1, 1))

            node.modelView.copyFrom(cam.view)
            node.modelView.mult(node.world)

            node.mvp.copyFrom(node.projection)
            node.mvp.mult(node.modelView)

            val instanceUbo = UBO(device, backingBuffer = stagingBuffer)
            instanceUbo.fromInstance(node)
            instanceUbo.createUniformBuffer(memoryProperties)
            instanceUbo.populate()
        }

        stagingBuffer.copyFromStagingBuffer()

        val instanceBuffer = if (state.vertexBuffers.containsKey("instance") && state.vertexBuffers["instance"]!!.size >= instanceBufferSize) {
            state.vertexBuffers["instance"]!!
        } else {
            logger.debug("Instance buffer for ${parentNode.name} needs to be required, insufficient size ($instanceBufferSize vs ${state.vertexBuffers["instance"]!!.size})")
            state.vertexBuffers["instance"]?.close()

            val buffer = VU.createBuffer(device,
                this.memoryProperties,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                wantAligned = true,
                allocationSize = instanceBufferSize * 1L)

            state.vertexBuffers.put("instance", buffer)
            buffer
        }

        with(VU.newCommandBuffer(device, commandPools.Standard, autostart = true)) {
            val copyRegion = VkBufferCopy.calloc(1)
                .size(instanceBufferSize * 1L)

            vkCmdCopyBuffer(this,
                stagingBuffer.buffer,
                instanceBuffer.buffer,
                copyRegion)

            copyRegion.free()
            this.endCommandBuffer(device, commandPools.Standard, queue, flush = true, dealloc = true)
        }

        state.instanceCount = instances.size

        stagingBuffer.close()
        return state
    }

    private fun createDescriptorPool(device: VkDevice): Long {
        // We need to tell the API the number of max. requested descriptors per type
        val typeCounts = VkDescriptorPoolSize.calloc(4)
        typeCounts[0]
            .type(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
            .descriptorCount(this.MAX_TEXTURES)

        typeCounts[1]
            .type(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC)
            .descriptorCount(this.MAX_UBOS)

        typeCounts[2]
            .type(VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT)
            .descriptorCount(this.MAX_INPUT_ATTACHMENTS)

        typeCounts[3]
            .type(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
            .descriptorCount(this.MAX_UBOS)

        // Create the global descriptor pool
        // All descriptors used in this example are allocated from this pool
        val descriptorPoolInfo = VkDescriptorPoolCreateInfo.calloc()
            .sType(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO)
            .pNext(NULL)
            .pPoolSizes(typeCounts)
            .maxSets(this.MAX_TEXTURES + this.MAX_UBOS + this.MAX_INPUT_ATTACHMENTS + this.MAX_UBOS)// Set the max. number of sets that can be requested

        val descriptorPool = VU.run(memAllocLong(1), "vkCreateDescriptorPool",
            function = { vkCreateDescriptorPool(device, descriptorPoolInfo, null, this) },
            cleanup = { descriptorPoolInfo.free(); typeCounts.free() })

        return descriptorPool
    }

    private fun prepareDefaultBuffers(device: VkDevice): HashMap<String, VulkanBuffer> {
        val map = HashMap<String, VulkanBuffer>()

        map.put("UBOBuffer", VU.createBuffer(device, this.memoryProperties,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            wantAligned = true,
            allocationSize = 512 * 1024 * 10))

        map.put("LightParametersBuffer", VU.createBuffer(device, this.memoryProperties,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            wantAligned = true,
            allocationSize = 512 * 1024 * 10))

        return map
    }

    private fun prepareDefaultUniformBuffers(device: VkDevice): ConcurrentHashMap<String, UBO> {
        val ubos = ConcurrentHashMap<String, UBO>()
        val defaultUbo = UBO(device)

        defaultUbo.name = "default"
        defaultUbo.members.put("ViewMatrix", { GLMatrix.getIdentity() })
        defaultUbo.members.put("ModelMatrix", { GLMatrix.getIdentity() })
        defaultUbo.members.put("ProjectionMatrix", { GLMatrix.getIdentity() })
        defaultUbo.members.put("MVP", { GLMatrix.getIdentity() })
        defaultUbo.members.put("CamPosition", { GLVector(0.0f, 0.0f, 0.0f) })
        defaultUbo.members.put("isBillboard", { 0 })

        defaultUbo.createUniformBuffer(memoryProperties)
        ubos.put("default", defaultUbo)

        val lightUbo = UBO(device)

        lightUbo.name = "BlinnPhongLighting"
        lightUbo.members.put("Position", { GLVector(0.0f, 0.0f, 0.0f) })
        lightUbo.members.put("La", { GLVector(0.0f, 0.0f, 0.0f) })
        lightUbo.members.put("Ld", { GLVector(0.0f, 0.0f, 0.0f) })
        lightUbo.members.put("Ls", { GLVector(0.0f, 0.0f, 0.0f) })

        lightUbo.createUniformBuffer(memoryProperties)
        ubos.put("BlinnPhongLighting", lightUbo)

        val materialUbo = UBO(device)

        materialUbo.name = "BlinnPhongMaterial"
        materialUbo.members.put("Ka", { GLVector(0.0f, 0.0f, 0.0f) })
        materialUbo.members.put("Kd", { GLVector(0.0f, 0.0f, 0.0f) })
        materialUbo.members.put("Ks", { GLVector(0.0f, 0.0f, 0.0f) })
        materialUbo.members.put("Shininess", { 1.0f })
        materialUbo.members.put("materialType", { 0 })

        materialUbo.createUniformBuffer(memoryProperties)
        ubos.put("BlinnPhongMaterial", materialUbo)

        return ubos
    }

    private fun recordSceneRenderCommands(device: VkDevice, pass: VulkanRenderpass, commandBuffer: VulkanCommandBuffer) {
        val target = pass.getOutput()

        logger.debug("Creating scene command buffer for {}/{} ({} attachments)", pass.name, target, target.attachments.count())

        pass.vulkanMetadata.renderPassBeginInfo
            .sType(VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO)
            .pNext(NULL)
            .renderPass(target.renderPass.get(0))
            .framebuffer(target.framebuffer.get(0))
            .renderArea(pass.vulkanMetadata.renderArea)
            .pClearValues(pass.vulkanMetadata.clearValues)

        val renderOrderList = ArrayList<Node>()

        scene.discover(scene, { n -> n is Renderable && n is HasGeometry && n.visible }).forEach {
            if(it.dirty) {
                if(it is FontBoard) {
                    updateFontBoard(it)
                }

                it.dirty = false
            }

            if(!it.metadata.containsKey("VulkanRenderer")) {
                logger.info("${it.name} is not initialized, doing that now")
                it.metadata.put("VulkanRenderer", VulkanObjectState())
                initializeNode(it)
            } else {
                renderOrderList.add(it)
            }
        }

        val instanceGroups = renderOrderList.groupBy(Node::instanceOf)

        // start command buffer recording
        if (commandBuffer.commandBuffer == null) {
            commandBuffer.commandBuffer = VU.newCommandBuffer(device, commandPools.Render, autostart = true)
        } else {
            vkResetCommandBuffer(commandBuffer.commandBuffer!!, VK_FLAGS_NONE)
            VU.beginCommandBuffer(commandBuffer.commandBuffer!!)
        }

        with(commandBuffer.commandBuffer) {

            vkCmdBeginRenderPass(this, pass.vulkanMetadata.renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE)

            vkCmdSetViewport(this, 0, pass.vulkanMetadata.viewport)
            vkCmdSetScissor(this, 0, pass.vulkanMetadata.scissor)

            instanceGroups[null]?.forEach nonInstancedDrawing@ { node ->
                val s = node.metadata["VulkanRenderer"]!! as VulkanObjectState

                if (node in instanceGroups.keys || s.vertexCount == 0) {
                    return@nonInstancedDrawing
                }

                pass.vulkanMetadata.vertexBufferOffsets.put(0, 0)
                pass.vulkanMetadata.vertexBuffers.put(0, s.vertexBuffers["vertex+index"]!!.buffer)
                pass.vulkanMetadata.descriptorSets.put(0, descriptorSets["default"]!!)

                if (s.textures.size > 0) {
                    pass.vulkanMetadata.descriptorSets.put(1, s.textureDescriptorSet)
                }

                val pipeline = pass.pipelines.getOrDefault("preferred-${node.name}", pass.pipelines["default"]!!)
                    .getPipelineForGeometryType((node as HasGeometry).geometryType)

                vkCmdBindPipeline(this, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline)
                vkCmdBindDescriptorSets(this, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    pipeline.layout, 0, pass.vulkanMetadata.descriptorSets, sceneUBOs[node]!!.offsets)
                vkCmdBindVertexBuffers(this, 0, pass.vulkanMetadata.vertexBuffers, pass.vulkanMetadata.vertexBufferOffsets)

                logger.trace("now drawing {}, {} DS bound, {} textures", node.name, pass.vulkanMetadata.descriptorSets.capacity(), s.textures.count())

                if (s.isIndexed) {
                    vkCmdBindIndexBuffer(this, pass.vulkanMetadata.vertexBuffers.get(0), s.indexOffset * 1L, VK_INDEX_TYPE_UINT32)
                    vkCmdDrawIndexed(this, s.indexCount, 1, 0, 0, 0)
                } else {
                    vkCmdDraw(this, s.vertexCount, 1, 0, 0)
                }
            }

            instanceGroups.keys.filterNotNull().forEach instancedDrawing@ { node ->
                val s = node.metadata["VulkanRenderer"]!! as VulkanObjectState

                // this only lets non-instanced, parent nodes through
                if (s.vertexCount == 0) {
                    return@instancedDrawing
                }

                pass.vulkanMetadata.vertexBufferOffsets.put(0, 0)
                pass.vulkanMetadata.vertexBuffers.put(0, s.vertexBuffers["vertex+index"]!!.buffer)
                pass.vulkanMetadata.descriptorSets.put(0, descriptorSets["default"]!!)

                if (s.textures.size > 0) {
                    pass.vulkanMetadata.descriptorSets.put(1, s.textureDescriptorSet)
                }

                pass.vulkanMetadata.instanceBuffers.put(0, s.vertexBuffers["instance"]!!.buffer)

                val pipeline = pass.pipelines.getOrElse("preferred-${node.name}",
                    {
                        logger.warn("Preferred pipeline for instanced node {} not found. Using default.", node.name)
                        pass.pipelines["default"]!!
                    }).getPipelineForGeometryType((node as HasGeometry).geometryType)

                vkCmdBindPipeline(this, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline)
                vkCmdBindDescriptorSets(this, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    pipeline.layout, 0, pass.vulkanMetadata.descriptorSets, sceneUBOs[node]!!.offsets)

                vkCmdBindVertexBuffers(this, 0, pass.vulkanMetadata.vertexBuffers, pass.vulkanMetadata.vertexBufferOffsets)
                vkCmdBindVertexBuffers(this, 1, pass.vulkanMetadata.instanceBuffers, pass.vulkanMetadata.vertexBufferOffsets)

                if (s.isIndexed) {
                    vkCmdBindIndexBuffer(this, pass.vulkanMetadata.vertexBuffers.get(0), s.indexOffset * 1L, VK_INDEX_TYPE_UINT32)
                    vkCmdDrawIndexed(this, s.indexCount, s.instanceCount, 0, 0, 0)
                } else {
                    vkCmdDraw(this, s.vertexCount, s.instanceCount, 0, 0)
                }
            }

            vkCmdEndRenderPass(this)
            this!!.endCommandBuffer()
        }
    }

    private fun recordPostprocessRenderCommands(device: VkDevice, pass: VulkanRenderpass, commandBuffer: VulkanCommandBuffer) {
        val target = pass.getOutput()

        logger.debug("Creating postprocessing command buffer for {}/{} ({} attachments)", pass.name, target, target.attachments.count())

        pass.vulkanMetadata.renderPassBeginInfo
            .sType(VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO)
            .pNext(NULL)
            .renderPass(target.renderPass.get(0))
            .framebuffer(target.framebuffer.get(0))
            .renderArea(pass.vulkanMetadata.renderArea)
            .pClearValues(pass.vulkanMetadata.clearValues)

        // start command buffer recording
        if (commandBuffer.commandBuffer == null) {
            commandBuffer.commandBuffer = VU.newCommandBuffer(device, commandPools.Render, autostart = true)
        } else {
            vkResetCommandBuffer(commandBuffer.commandBuffer!!, VK_FLAGS_NONE)
            VU.beginCommandBuffer(commandBuffer.commandBuffer!!)
        }

        with(commandBuffer.commandBuffer) {

            vkCmdBeginRenderPass(this, pass.vulkanMetadata.renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE)

            vkCmdSetViewport(this, 0, pass.vulkanMetadata.viewport)
            vkCmdSetScissor(this, 0, pass.vulkanMetadata.scissor)

            val pipeline = pass.pipelines["default"]!!
            val vulkanPipeline = pipeline.getPipelineForGeometryType(GeometryType.TRIANGLES)

            if(pass.vulkanMetadata.descriptorSets.capacity() < pipeline.descriptorSpecs.count()) {
                memFree(pass.vulkanMetadata.descriptorSets)
                pass.vulkanMetadata.descriptorSets = memAllocLong(pipeline.descriptorSpecs.count())
            }

            // allocate more vertexBufferOffsets than needed, set limit lateron
            pass.vulkanMetadata.uboOffsets.limit(16)
            (0..15).forEach { pass.vulkanMetadata.uboOffsets.put(it, 0) }

            var requiredDynamicOffsets = 0
            if(logger.isDebugEnabled) {
                logger.debug("descriptor sets are {}", pass.descriptorSets.keys.joinToString(", "))
                logger.debug("pipeline provides {}", pipeline.descriptorSpecs.map { it.name }.joinToString(", "))
            }

            pipeline.descriptorSpecs.forEachIndexed { i, spec ->
                val dsName = if (spec.name.startsWith("ShaderParameters")) {
                    "ShaderParameters-${pass.name}"
                } else if (spec.name.startsWith("inputs")) {
                    "inputs-${pass.name}"
                } else if (spec.name.startsWith("Matrices")) {
                    pass.vulkanMetadata.uboOffsets.put(sceneUBOs.values.first().offsets)
                    requiredDynamicOffsets += 3

                    "default"
                } else {
                    if (spec.name.startsWith("LightParameters")) {
                        pass.vulkanMetadata.uboOffsets.put(0)
                        requiredDynamicOffsets++
                    }

                    spec.name
                }

                val set = if (dsName == "default" || dsName == "LightParameters") {
                    descriptorSets.get(dsName)
                } else {
                    pass.descriptorSets.get(dsName)
                }

                if (set != null) {
                    logger.debug("Adding DS#{} for {} to required pipeline DSs", i, dsName)
                    pass.vulkanMetadata.descriptorSets.put(i, set)
                } else {
                    logger.error("DS for {} not found!", dsName)
                }
            }

            // see if this stage requires dynamic buffers
            pass.vulkanMetadata.uboOffsets.limit(requiredDynamicOffsets)
            pass.vulkanMetadata.uboOffsets.position(0)

            vkCmdBindPipeline(this, VK_PIPELINE_BIND_POINT_GRAPHICS, vulkanPipeline.pipeline)
            vkCmdBindDescriptorSets(this, VK_PIPELINE_BIND_POINT_GRAPHICS,
                vulkanPipeline.layout, 0, pass.vulkanMetadata.descriptorSets, pass.vulkanMetadata.uboOffsets)

            vkCmdDraw(this, 3, 1, 0, 0)

            vkCmdEndRenderPass(this)
            this!!.endCommandBuffer()
        }
    }

    private fun updateInstanceBuffers() {
        val renderOrderList = ArrayList<Node>()

        scene.discover(scene, { n -> n is Renderable && n is HasGeometry && n.visible }).forEach {
            renderOrderList.add(it)
        }

        val instanceGroups = renderOrderList.groupBy(Node::instanceOf)

        instanceGroups.keys.filterNotNull().forEach { node ->
            updateInstanceBuffer(device, node, node.metadata["VulkanRenderer"] as VulkanObjectState)
        }
    }

    @Synchronized private fun updateDefaultUBOs(device: VkDevice) {
        val cam = scene.findObserver()
        cam.view = cam.getTransformation()

        buffers["UBOBuffer"]!!.reset()

        sceneUBOs.forEach { node, ubo ->
            node.updateWorld(true, false)

            if(ubo.offsets == null || ubo.offsets!!.capacity() < 3) {
                ubo.offsets = memAllocInt(3)
            }

            (0..2).forEach { ubo.offsets!!.put(it, 0) }

            var bufferOffset = buffers["UBOBuffer"]!!.advance()
            ubo.offsets!!.put(0, bufferOffset)

            node.projection.copyFrom(cam.projection)
            node.projection.set(1, 1, -1.0f * cam.projection.get(1, 1))

            node.view.copyFrom(cam.view)

            node.mvp.copyFrom(node.projection)
            node.mvp.mult(node.modelView)

            ubo.populate(offset = bufferOffset.toLong())

            val materialUbo = (node.metadata["VulkanRenderer"]!! as VulkanObjectState).UBOs["BlinnPhongMaterial"]!!
            bufferOffset = buffers["UBOBuffer"]!!.advance()
            ubo.offsets!!.put(1, bufferOffset)

            materialUbo.populate(offset = bufferOffset.toLong())
        }

        buffers["UBOBuffer"]!!.copyFromStagingBuffer()

        buffers["LightParametersBuffer"]!!.reset()

        val lights = scene.discover(scene, { n -> n is PointLight })

        val lightUbo = UBO(device, backingBuffer = buffers["LightParametersBuffer"]!!)
        lightUbo.members.put("numLights", { lights.size })
        lightUbo.members.put("filler1", { 0.0f })
        lightUbo.members.put("filler2", { 0.0f })
        lightUbo.members.put("filler3", { 0.0f })

        lights.forEachIndexed { i, light ->
            val l = light as PointLight
            l.updateWorld(true, false)

            lightUbo.members.put("Linear-$i", { l.linear })
            lightUbo.members.put("Quadratic-$i", { l.quadratic })
            lightUbo.members.put("Intensity-$i", { l.intensity })
            lightUbo.members.put("Position-$i", { l.position })
            lightUbo.members.put("Color-$i", { l.emissionColor })
            lightUbo.members.put("filler-$i", { 0.0f })
        }

        lightUbo.createUniformBuffer(memoryProperties)
        lightUbo.populate()

        buffers["LightParametersBuffer"]!!.copyFromStagingBuffer()
    }

    override fun screenshot() {
        screenshotRequested = true
    }

    fun Int.toggle(): Int {
        if (this == 0) {
            return 1
        } else if (this == 1) {
            return 0
        }

        logger.warn("Property is not togglable.")
        return this
    }

    fun toggleDebug() {
        settings.getAllSettings().forEach {
            if (it.toLowerCase().contains("debug")) {
                try {
                    val property = settings.get<Int>(it).toggle()
                    settings.set(it, property)

                } catch(e: Exception) {
                    logger.warn("$it is a property that is not togglable.")
                }
            }
        }
    }

    override fun close() {
        logger.info("Renderer teardown started.")

        logger.debug("Closing nodes...")
        textureCache.forEach { it.value.close() }
        scene.discover(scene, { n -> n is Renderable }).forEach {
            destroyNode(it)
        }

        logger.debug("Closing buffers...")
        buffers.forEach { s, vulkanBuffer -> vulkanBuffer.close() }
        standardUBOs.forEach { it.value.close() }

        vertexDescriptors.forEach {
            it.value.attributeDescription?.free()
            it.value.bindingDescription?.free()
            it.value.state.free()
        }

        logger.debug("Closing descriptor sets and pools...")
        descriptorSetLayouts.forEach { vkDestroyDescriptorSetLayout(device, it.value, null) }
        vkDestroyDescriptorPool(device, descriptorPool, null)

        logger.debug("Closing command buffers...")
        ph.commandBuffers.free()
        memFree(ph.signalSemaphore)
        memFree(ph.waitSemaphore)
        memFree(ph.waitStages)

        semaphores.forEach { it.value.forEach { semaphore -> vkDestroySemaphore(device, semaphore, null) } }

        memFree(firstWaitSemaphore)
        semaphoreCreateInfo.free()

        logger.debug("Closing swapchain...")
        memFree(pSwapchains)
        memFree(swapchainImage)

        swapchain?.let {
            it.close()
            vkDestroySwapchainKHR(device, it.handle, null)
            vkDestroySurfaceKHR(instance, surface, null)
        }

        logger.debug("Closing renderpasses...")
        renderpasses.forEach { s, vulkanRenderpass ->
            vulkanRenderpass.close()
        }

        with(commandPools) {
            destroyCommandPool(device, Render)
            destroyCommandPool(device, Compute)
            destroyCommandPool(device, Standard)
        }

        vkDestroyPipelineCache(device, pipelineCache, null)

        if(validation) {
            vkDestroyDebugReportCallbackEXT(instance, debugCallbackHandle, null)
            debugCallback.free()
        }
        layers.forEach { memFree(it) }

        logger.debug("Closing device...")
        vkDeviceWaitIdle(device)
        vkDestroyDevice(device, null)
        logger.debug("Closing instance...")
        vkDestroyInstance(instance, null)

        memoryProperties.free()

        windowSizeCallback.close()
        glfwDestroyWindow(window.glfwWindow!!)

        logger.info("Renderer teardown complete.")
    }

    override fun reshape(newWidth: Int, newHeight: Int) {

    }

    fun toggleFullscreen() {
        toggleFullscreen = !toggleFullscreen
    }

    fun switchFullscreen() {
        if(window.isFullscreen) {
            glfwSetWindowMonitor(window.glfwWindow!!,
                NULL,
                0, 0,
                window.width, window.height, GLFW_DONT_CARE)
            glfwSetWindowPos(window.glfwWindow!!, 100, 100)

            swapchainRecreator.mustRecreate = true
            window.isFullscreen = false
        } else {
            val preferredMonitor = System.getProperty("scenery.FullscreenMonitor", "0").toInt()

            val monitor = if(preferredMonitor == 0) {
                glfwGetPrimaryMonitor()
            } else {
                val monitors = glfwGetMonitors()
                if(monitors.remaining() < preferredMonitor) {
                    monitors.get(0)
                } else {
                    monitors.get(preferredMonitor)
                }
            }

            glfwSetWindowMonitor(window.glfwWindow!!,
                monitor,
                0, 0,
                window.width, window.height, GLFW_DONT_CARE)

            swapchainRecreator.mustRecreate = true
            window.isFullscreen = true
        }
    }
}
