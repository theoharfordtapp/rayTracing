# AQA A-Level Computer Science NEA 2024
#
# Graphics Engine
# Theo HT
#
# (Explorer)
#
# NOTE # Comments with `MARK` in them are purely IDE-related. They have no relevance to the code.

# MARK: IMPORTS
import copy
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QFileDialog, QListWidget,
    QLineEdit, QTabWidget, QSizePolicy, QProgressDialog, QMessageBox, QScrollArea, QAction, QMenu, QColorDialog, QListWidgetItem
)
from T3DE import Scene, RGB, Shader, Vec3, Euler, Scale, Camera, Light, Sphere, Cube, Object, Mesh
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QPixmap, QCursor, QColor
from T3DR import Texel, Photon, Debug, Post
from string import ascii_uppercase
from threading import Lock
import qimage2ndarray
import numpy as np
import math
import cv2
import sys


# MARK: SHARED STATE
## A shared set of global information for different threads to access
class SharedState:
    def __init__(self) -> None:
        ## Lock prevents race conditions
        ## Using `with self.lock` will lock the SharedState while the with block is being executed
        self.lock = Lock()
        self.__sceneChanged = True
        self.__selectedObject = -1
        self.__normals = False
        self.__engine = 'texel'
        self.__renderCancelled = False
        self.__vertices = False
        self.__selectedFaces = []
        self.__texelWidth = 1440
        self.__texelHeight = 900
    
    # MARK: > SCENE CHANGED
    @property
    def sceneChanged(self):
        with self.lock:
            return self.__sceneChanged

    @sceneChanged.setter
    def sceneChanged(self, sceneChanged):
        with self.lock:
            self.__sceneChanged = sceneChanged
    
    # MARK: > WIDTH
    @property
    def texelWidth(self):
        with self.lock:
            return self.__texelWidth

    @texelWidth.setter
    def texelWidth(self, texelWidth):
        with self.lock:
            self.__texelWidth = texelWidth
    
    # MARK: > HEIGHT
    @property
    def texelHeight(self):
        with self.lock:
            return self.__texelHeight

    @texelHeight.setter
    def texelHeight(self, texelHeight):
        with self.lock:
            self.__texelHeight = texelHeight
    
    # MARK: > SCENE CHANGED
    @property
    def selectedFaces(self):
        with self.lock:
            return self.__selectedFaces

    @selectedFaces.setter
    def selectedFaces(self, selectedFaces):
        with self.lock:
            self.__selectedFaces = selectedFaces
    
    def selectedFacesAppend(self, val):
        with self.lock:
            self.__selectedFaces.append(val)
    
    def selectedFacesRemove(self, val):
        with self.lock:
            self.__selectedFaces.remove(val)
    
    # MARK: > VERTICESW
    @property
    def vertices(self):
        with self.lock:
            return self.__vertices

    @vertices.setter
    def vertices(self, vertices):
        with self.lock:
            self.__vertices = vertices
    
    # MARK: > SELECTED OBJ
    @property
    def selectedObject(self):
        with self.lock:
            return self.__selectedObject

    @selectedObject.setter
    def selectedObject(self, selectedObject):
        with self.lock:
            self.__selectedObject = selectedObject

    # MARK: > NORMALS
    @property
    def normals(self):
        with self.lock:
            return self.__normals

    @normals.setter
    def normals(self, normals):
        with self.lock:
            self.__normals = normals

    # MARK: > ENGINE
    @property
    def engine(self):
        with self.lock:
            return self.__engine

    @engine.setter
    def engine(self, engine):
        with self.lock:
            self.__engine = engine

    # MARK: > ENGINE
    @property
    def renderCancelled(self):
        with self.lock:
            return self.__renderCancelled

    @renderCancelled.setter
    def renderCancelled(self, renderCancelled):
        with self.lock:
            self.__renderCancelled = renderCancelled


# MARK: CLICKABLE
## Clickable image widet for the viewport background
class ClickableImage(QLabel):
    ## Setup clicked signal
    clicked = pyqtSignal(int, int)
    
    def mousePressEvent(self, event):
        ## Send the signal back to the connected slot
        self.clicked.emit(event.x(), event.y())


# MARK: PANEL
## Collapsible panel widget
class CollapsiblePanel(QWidget):
    def __init__(self, title, parent=None) -> None:
        super().__init__(parent)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.title = title

        self.toggle = QPushButton('> ' + self.title)
        # self.toggle.setCheckable(True)
        self.toggle.clicked.connect(self.togglePanel)
        self.layout.addWidget(self.toggle)

        self.panel = QWidget()
        self.panelLayout = QVBoxLayout(self.panel)
        self.panelLayout.setContentsMargins(0, 0, 0, 0)
        self.panelLayout.setAlignment(Qt.AlignTop)
        self.layout.addWidget(self.panel)
        
        self.panel.setStyleSheet('CollapsiblePanel > QWidget { background-color: #e3e3e3; border-top-left-radius: 0; border-top-right-radius: 0; border-bottom-left-radius: 5px; border-bottom-right-radius: 5px; }')
        self.toggle.setStyleSheet('CollapsiblePanel > QPushButton { padding: 3px; background-color: #ffffff; border-radius: 5px; } CollapsiblePanel > QPushButton:hover { background-color: #ffe1e3; }')

        self.expanded = False

        # self.panel.setFixedHeight(self.panel.sizeHint().height())
        self.panel.setFixedHeight(0)

    # MARK: > TOGGLE
    ## Toggle panel visiblity
    def togglePanel(self):
        # if self.toggle.isChecked():
        if not self.expanded:
            self.toggle.setStyleSheet('CollapsiblePanel > QPushButton { background-color: #e3e3e3; border-top-left-radius: 5px; border-top-right-radius: 5px; border-bottom-left-radius: 0; border-bottom-right-radius: 0; padding-top: 3px; padding-bottom: 1px; } CollapsiblePanel > QPushButton:hover { background-color: #efdfe1; }')
            self.toggle.setText('V ' + self.title)
            self.expanded = True
            self.panel.setFixedHeight(self.panel.sizeHint().height())
        else:
            # self.toggle.setStyleSheet('padding: 3px; background-color: #ffffff; border-radius: 5px;')
            self.toggle.setStyleSheet('CollapsiblePanel > QPushButton { padding: 3px; background-color: #ffffff; border-radius: 5px; } CollapsiblePanel > QPushButton:hover { background-color: #ffe1e3; }')
            self.toggle.setText('> ' + self.title)
            self.expanded = False
            self.panel.setFixedHeight(0)


# MARK: > DRAG BUTTON
class DragButton(QPushButton):
    onDrag = pyqtSignal(int, int)
    
    def __init__(self, text, parent=None):
        super().__init__(text, parent=parent)
        self.originalPos = None
        self.offset = 0
        self.setMouseTracking(True)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            self.originalPos = QCursor.pos()
            QApplication.setOverrideCursor(Qt.BlankCursor)
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, _):
        if QApplication.overrideCursor() and self.originalPos:
            currentPos = QCursor.pos()
            self.onDrag.emit(currentPos.x() - self.originalPos.x(), currentPos.y() - self.originalPos.y())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            QApplication.restoreOverrideCursor()
            buttonCenter = self.mapToGlobal(self.rect().center())
            QCursor.setPos(buttonCenter)
        super().mouseReleaseEvent(event)


# MARK: VIEWPORT THREAD
## Separate thread for rendering viewport concurrently with the GUI
class ViewportThread(QThread):
    ## Setup render complete signal
    renderComplete = pyqtSignal(np.ndarray)
    
    def __init__(self, main, sharedState, parent=None):
        super().__init__(parent)
        self.sharedState = sharedState
        self.texel = main.texel
        self.debug = main.debug

    # MARK: > RUN
    ## Render the scene continuously
    def run(self):
        while True:
            ## Only run if scene has changed
            if self.sharedState.sceneChanged:
                ## Change the selected object gizmo
                self.texel.options['selectedObject'] = self.sharedState.selectedObject
                self.texel.options['normals'] = self.sharedState.normals
                self.texel.options['vertices'] = self.sharedState.vertices
                self.texel.options['selectedFaces'] = self.sharedState.selectedFaces

                self.texel.options['width'] = self.sharedState.texelWidth
                self.texel.options['height'] = self.sharedState.texelHeight

                engine = self.texel if self.sharedState.engine == 'texel' else self.debug
                
                ## Get image data from render
                render = engine.render()
                renderedData = render.imageAs('np')
                # renderedData = Post.upscale(renderedData, 10)

                ## Send the signal back to the connected slot
                self.renderComplete.emit(renderedData)
                
                self.sharedState.sceneChanged = False

                # ## Add delay between renders to balance performance
                # self.msleep(1000)


# MARK: RENDER THREAD
## Separate thread for rendering final render
class RenderThread(QThread):
    ## Setup progress update signal
    progressUpdate = pyqtSignal(dict, int, int)
    ## Setup render complete signal
    renderComplete = pyqtSignal(np.ndarray)
    
    def __init__(self, photon, sharedState, parent=None):
        super().__init__(parent)
        self.photon = photon
        self.sharedState = sharedState
    
    def run(self):
        render = self.photon.render(progressCallback=lambda data, key, total: self.progressUpdate.emit(data, key, total), cancelCallback=lambda: self.sharedState.renderCancelled, returnType='rendered')
        print('received rendered')
        renderedData = render.imageAs('np')
        print('extracted np')
        # renderedData = self.photon.render(progressCallback=lambda data, key, total: self.progressUpdate.emit(data, key, total), cancelCallback=lambda: self.sharedState.renderCancelled, returnType='np')
        
        # renderedData = Post.upscale(renderedData, 6)
        
        self.renderComplete.emit(renderedData)


# MARK: MAIN WINDOW
## Main application window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        ## Setup shared state for global information
        self.sharedState = SharedState()
        
        ## Setup scene for explorer
        self.scene = Scene()

        ## Setup photon instance for ray-tracing to detect clicks
        self.tracer = Photon(scene=self.scene, options={
            'aabb': True,
            'bvh': True,
            'debug': False,
            'transformBefore': False,
        })
        
        ## Setup rendering engines
        self.renderer = Photon(scene=self.scene, options={
            'width': 72,
            'height': 45,
        
            'bounces': 2,
            'samples': 1,
            
            'aabb': True,
            'bvh': True,
            
            'wait': False,
            'debug': False,
            
            'lights': True,
            
            'ambient': RGB(32, 32, 32),
            
            'threads': 1,
        })
        self.texel = Texel(scene=self.scene, options={
            'width': 1440,
            'height': 900,
            'edges': True,
            'lights': True,
            'vertices': True,
            'axes': False,
            'ambient': RGB(32, 32, 32),
            'selectedObject': self.sharedState.selectedObject,
            'normals': self.sharedState.normals,
            'attenuation': True,
            'lighting': True,
        })
        self.debug = Debug(scene=self.scene, options={
            'direction': 'side',
            'size': 3
        })
        
        self.newScene()

        ## Keep track of most recent render
        self.lastRender = None
        
        self.facesMode = False
        self.shiftHeld = False

        ## Setup window
        self.setWindowTitle('3D Explorer')
        self.setGeometry(100, 100, 800, 600)

        ## Setup central widget
        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.layout = QHBoxLayout()
        self.centralWidget.setLayout(self.layout)
        
        ## Setup menu bar
        menuBar = self.menuBar()

        ## Setup file menu
        fileMenu = menuBar.addMenu("File")

        ## Setup new action
        newAction = QAction("New", self)
        newAction.triggered.connect(self.newScene)
        fileMenu.addAction(newAction)

        ## Setup open action
        openAction = QAction("Open", self)
        openAction.triggered.connect(self.openScene)
        fileMenu.addAction(openAction)

        ## Setup save action
        saveAction = QAction("Save", self)
        saveAction.triggered.connect(self.saveScene)
        fileMenu.addAction(saveAction)

        ## Setup render menu
        renderMenu = menuBar.addMenu("Render")

        ## Setup render action
        renderAction = QAction("Render", self)
        renderAction.triggered.connect(self.renderPhoton)
        renderMenu.addAction(renderAction)

        ## Setup save render action
        saveRenderAction = QAction("Save Render", self)
        saveRenderAction.triggered.connect(self.saveRender)
        renderMenu.addAction(saveRenderAction)

        ## Setup object menu
        objectMenu = menuBar.addMenu("Object")

        ## Setup add menu
        addObjectMenu = QMenu("Add", self)
        objectMenu.addMenu(addObjectMenu)

        ## Setup add cube action
        addCubeAction = QAction("Cube", self)
        addCubeAction.triggered.connect(lambda: self.addObject('cube'))
        addObjectMenu.addAction(addCubeAction)

        ## Setup add light action
        addLightAction = QAction("Light", self)
        addLightAction.triggered.connect(lambda: self.addObject('light'))
        addObjectMenu.addAction(addLightAction)

        ## Setup add sphere action
        addSphereAction = QAction("Sphere", self)
        addSphereAction.triggered.connect(lambda: self.addObject('sphere'))
        addObjectMenu.addAction(addSphereAction)

        ## Setup add empty action
        addEmptyAction = QAction("Empty", self)
        addEmptyAction.triggered.connect(lambda: self.addObject('empty'))
        addObjectMenu.addAction(addEmptyAction)

        ## Setup add mesh action
        addMeshAction = QAction("Mesh", self)
        addMeshAction.triggered.connect(lambda: self.addObject('mesh'))
        addObjectMenu.addAction(addMeshAction)

        ## Setup viewport
        self.viewport = ClickableImage('...')
        self.viewport.clicked.connect(self.handleViewportClick)
        self.viewport.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.viewport)

        ## Setup scrollable controls area
        self.controlsScrollArea = QScrollArea()
        self.controlsScrollArea.setWidgetResizable(True)
        
        # Setup controls container
        self.controlsContainer = QWidget()
        self.controlsScrollArea.setWidget(self.controlsContainer)
        self.controlsLayout = QVBoxLayout(self.controlsContainer)
        self.controlsLayout.setAlignment(Qt.AlignTop)
        
        # Add the scroll area to the main layout
        self.layout.addWidget(self.controlsScrollArea)

        ## Setup button to toggle normals visibility
        self.normalsButton = QPushButton('Normals')
        self.normalsButton.setCheckable(True)
        self.normalsButton.clicked.connect(self.toggleNormals)
        self.controlsLayout.addWidget(self.normalsButton)

        ## Setup button to flip normals of selected object
        self.flipNormalsButton = QPushButton('Flip normals')
        self.flipNormalsButton.clicked.connect(self.flipSelectedNormals)
        self.controlsLayout.addWidget(self.flipNormalsButton)

        ## Setup button to delete object
        self.deleteObjectButton = QPushButton('Delete')
        self.deleteObjectButton.clicked.connect(self.deleteObject)
        self.controlsLayout.addWidget(self.deleteObjectButton)

        ## Setup render panel
        self.renderPanel = CollapsiblePanel('Render')
        self.renderPanel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.controlsLayout.addWidget(self.renderPanel)
        self.setupRenderPanel()

        ## Setup viewport panel
        self.viewportPanel = CollapsiblePanel('Viewport')
        self.viewportPanel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.controlsLayout.addWidget(self.viewportPanel)
        self.setupViewportPanel()

        ## Setup camera panel
        self.cameraPanel = CollapsiblePanel('Camera')
        self.cameraPanel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.controlsLayout.addWidget(self.cameraPanel)
        self.setupCameraPanel()

        ## Setup object panel
        self.objectPanel = CollapsiblePanel('Object')
        self.objectPanel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.controlsLayout.addWidget(self.objectPanel)
        self.setupObjectPanel()
        
        ## Setup shader panel
        self.shaderPanel = CollapsiblePanel('Shader')
        self.shaderPanel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.controlsLayout.addWidget(self.shaderPanel)
        self.setupShaderPanel()

        ## Setup light panel
        self.lightPanel = CollapsiblePanel('Light')
        self.lightPanel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.controlsLayout.addWidget(self.lightPanel)
        self.setupLightPanel()

        ## Setup transform panel
        self.transformPanel = CollapsiblePanel('Transform')
        self.transformPanel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.controlsLayout.addWidget(self.transformPanel)
        self.setupTransformPanel()

        self.originalTransforms = [Vec3(0, 0, 0), Euler(0, 0, 0), Scale(1, 1, 1)]

        ## Setup threading for separate viewport render
        self.viewportThread = ViewportThread(main=self, sharedState=self.sharedState)
        self.viewportThread.renderComplete.connect(self.displayRenderedImage)

        self.refreshTransformProperties()
        self.refreshRenderProperties()
        self.refreshCameraProperties()
        self.refreshShaderProperties()
        self.viewportThread.start()
    
    def keyPressEvent(self, event):
        if event.modifiers() & Qt.ShiftModifier:
            self.shiftHeld = True
        
        if event.key() == Qt.Key_F:
            self.toggleFacesMode()

        if event.key() == Qt.Key_Backspace:
            self.deleteObject()

    def keyReleaseEvent(self, event):
        if not event.modifiers() & Qt.ShiftModifier:
            self.shiftHeld = False
    
    def addObject(self, typeOf):
        if typeOf == 'light':
            obj = Light(scene=self.scene)
        elif typeOf == 'sphere':
            obj = Sphere(scene=self.scene)
        elif typeOf == 'cube':
            obj = Cube(scene=self.scene)
        elif typeOf == 'empty':
            obj = Object(scene=self.scene)
        elif typeOf == 'mesh':
            filepath = QFileDialog.getOpenFileName(self, "Load mesh", "", "STL (*.stl)")[0]
            mesh = Mesh.fromSTL(filepath)
            obj = Object(scene=self.scene, mesh=mesh)

        obj.bvh = obj.buildBVH()
        
        self.refreshTransformProperties()
        self.refreshObjectList()
        self.sharedState.sceneChanged = True
        
        print(self.scene.objects)
    
    def newScene(self):
        self.scene = Scene()
        Camera(self.scene, name='Camera', position=Vec3(0, 0, -8))
        Cube(self.scene, name='Cube', position=Vec3(0, 0, 0))
        Light(self.scene, name='Light', position=Vec3(0, 3, 0))

        self.tracer.scene = self.scene
        self.renderer.scene = self.scene
        self.texel.scene = self.scene
        self.debug.scene = self.scene
        
        self.sharedState.sceneChanged = True

    def openScene(self):
        filepath = QFileDialog.getOpenFileName(self, "Open Scene", "", "JSON (*.json)")[0]
        if filepath:
            self.scene = Scene.fromJSON(filepath)
            
            ## Build BVH for all objects
            for obj in self.scene.objects:
                obj.bvh = obj.buildBVH()
            
            self.tracer.scene = self.scene
            self.renderer.scene = self.scene
            self.texel.scene = self.scene
            self.debug.scene = self.scene
            
            self.sharedState.sceneChanged = True
            
            self.refreshShaderProperties()
            self.refreshTransformProperties()
            self.refreshRenderProperties()
            self.refreshLightProperties()
            self.refreshCameraProperties()
    
    def saveRender(self):
        filepath = QFileDialog.getSaveFileName(self, "Save Render", "", "PNG (*.png);;JPEG (*.jpg, *.jpeg)")[0]
        if filepath:
            cv2.imwrite(filepath, self.lastRender)
    
    def saveScene(self):
        # filepath = QFileDialog.getOpenFileName()[0]
        filepath = QFileDialog.getSaveFileName(self, "Save Scene", "", "JSON (*.json)")[0]
        if filepath:
            self.scene.saveJSON(filepath)
    
    def deleteObject(self):
        obj = self.scene.getObject(self.sharedState.selectedObject)
        
        if obj != None:
            self.sharedState.selectedObject = -1
            self.scene.objects.remove(obj)
            self.refreshShaderProperties()
            self.refreshTransformProperties()
            self.refreshRenderProperties()
            self.refreshLightProperties()
            self.sharedState.sceneChanged = True
    
    # MARK: > CASE FORMAT
    ## Convert camel case to words
    def camelToWords(self, string):
        returnString = ''
        
        for i, char in enumerate(string):
            if i == 0:
                returnString += char.upper()
            elif char in ascii_uppercase:
                returnString += ' ' + char.lower()
            else:
                returnString += char
        
        return returnString
    
    def setupShaderPanel(self):
        self.shaderList = QListWidget()
        self.shaderList.currentRowChanged.connect(self.refreshShaderProperties)
        # self.shaderList.currentRowChanged.connect(lambda: print(self.shaderList.currentRow()))
        self.shaderPanel.panelLayout.addWidget(self.shaderList)

        buttonLayout = QHBoxLayout()
        self.addShaderButton = QPushButton("Add")
        self.addShaderButton.clicked.connect(self.addShader)
        self.removeShaderButton = QPushButton("Remove")
        self.removeShaderButton.clicked.connect(self.removeShader)
        buttonLayout.addWidget(self.addShaderButton)
        buttonLayout.addWidget(self.removeShaderButton)
        self.shaderPanel.panelLayout.addLayout(buttonLayout)
        
        self.assignShaderButton = QPushButton("Assign")
        self.assignShaderButton.clicked.connect(self.assignShader)
        self.shaderPanel.panelLayout.addWidget(self.assignShaderButton)
        self.facesModeButton = QPushButton("Faces")
        self.facesModeButton.clicked.connect(self.toggleFacesMode)
        self.facesModeButton.setCheckable(True)
        self.shaderPanel.panelLayout.addWidget(self.facesModeButton)

        self.shaderPropertiesRegion = QWidget()
        self.shaderPropertiesRegion.setStyleSheet('background-color: #efefef; border-radius: 4px; margin: 4px;')
        self.shaderPropertiesRegion.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.shaderPropertiesRegionLayout = QVBoxLayout()
        self.shaderPropertiesRegionLayout.setAlignment(Qt.AlignTop)
        self.shaderPropertiesRegion.setLayout(self.shaderPropertiesRegionLayout)
        self.shaderPanel.panelLayout.addWidget(self.shaderPropertiesRegion)
            
        dummyShader = Shader()

        self.shaderPropertyInputs = {}

        for propertyName, propertyValue in vars(dummyShader).items():
            inputField = None
            typeOf = type(propertyValue)

            if typeOf in [str, float, int]:
                inputField = QLineEdit(self)
                inputField.setPlaceholderText(self.camelToWords(propertyName))
                inputField.setText(str(propertyValue))
                inputField.editingFinished.connect(self.updateShaderProperties)
                inputField.setStyleSheet('QLineEdit { background-color: #dedede; padding-top: 1px; padding-bottom: 1px; } QLineEdit:hover { background-color: #efdfe1; }')
            elif typeOf == bool:
                inputField = QPushButton(self.camelToWords(propertyName))
                inputField.setCheckable(True)
                if propertyValue:
                    inputField.setChecked(True)
                inputField.clicked.connect(self.updateShaderProperties)
                inputField.setStyleSheet('QPushButton { background-color: #ffffff; padding-top: 2px; padding-bottom: 2px; } QPushButton:hover { background-color: #ffe1e3; } QPushButton:checked { background-color: #f488b8 }')
            elif typeOf == RGB:
                inputField = QPushButton(self.camelToWords(propertyName))
                inputField.clicked.connect(lambda _, p=propertyName: self.pickShaderColor(p))
                inputField.setStyleSheet(f'QPushButton {{ background-color: rgb({propertyValue.r}, {propertyValue.g}, {propertyValue.b}); color: black; }}')

            if inputField:
                self.shaderPropertyInputs[propertyName] = inputField
                self.shaderPropertiesRegionLayout.addWidget(inputField)
    
        self.iorField = QLineEdit(self)
        self.iorField.setPlaceholderText('IOR')
        self.iorField.setText('1')
        self.iorField.editingFinished.connect(self.updateShaderProperties)
        self.iorField.setStyleSheet('QLineEdit { background-color: #dedede; padding-top: 1px; padding-bottom: 1px; } QLineEdit:hover { background-color: #efdfe1; }')
        self.shaderPanel.panelLayout.addWidget(self.iorField)
    
    def addShader(self):
        obj = self.scene.getObject(self.sharedState.selectedObject)
        if obj == None:
            return
        
        obj.shaders.append(Shader())
        
        self.refreshShaderProperties()
    
    def assignShader(self):
        obj = self.scene.getObject(self.sharedState.selectedObject)
        if obj == None:
            return
        
        currentRow = self.shaderList.currentRow()
        if currentRow < 0:
            return
        
        shader = obj.shaders[currentRow]
        obj.setShader(shader, self.sharedState.selectedFaces)
    
    def toggleFacesMode(self):
        if self.scene.getObject(self.sharedState.selectedObject) != None:
            self.facesMode = not self.facesMode
            self.sharedState.vertices = not self.sharedState.vertices
            self.sharedState.sceneChanged = True
        
        if not self.facesMode:
            self.sharedState.selectedFaces = []
        
        self.facesModeButton.setChecked(self.facesMode)
    
    def removeShader(self):
        obj = self.scene.getObject(self.sharedState.selectedObject)
        if obj == None:
            return

        if len(obj.shaders) == 1:
            return
        
        currentRow = self.shaderList.currentRow()
        if currentRow < 0:
            return
        
        shader = obj.shaders[currentRow]
        obj.shaders.remove(shader)
        
        
        newIndices = copy.deepcopy(obj.shaderIndices)
        for i, shaderIndex in enumerate(obj.shaderIndices):
            if shaderIndex == currentRow:
                newIndices[i] = 0
            elif shaderIndex > currentRow:
                newIndices[i] = shaderIndex-1
        
        obj.shaderIndices = newIndices
        
        self.refreshShaderProperties()
        self.sharedState.sceneChanged = True
    
    def pickShaderColor(self, propertyName):
        obj = self.scene.getObject(self.sharedState.selectedObject)
        if obj == None:
            return

        currentRow = self.shaderList.currentRow()
        if currentRow < 0:
            return

        shader = obj.shaders[currentRow]

        currentColor = getattr(shader, propertyName)
        initialColor = QColor(currentColor.r, currentColor.g, currentColor.b)
        newColor = QColorDialog.getColor(initialColor, self, self.camelToWords(propertyName))

        if newColor.isValid():
            newRGB = RGB(newColor.red(), newColor.green(), newColor.blue())
            setattr(shader, propertyName, newRGB)
            self.shaderPropertyInputs[propertyName].setStyleSheet(f'QPushButton {{ background-color: rgb({newRGB.r}, {newRGB.g}, {newRGB.b}); color: black; }}')
            self.sharedState.sceneChanged = True
        
    def updateShaderProperties(self):
        obj = self.scene.getObject(self.sharedState.selectedObject)
        if obj is None:
            self.refreshShaderProperties()
            return

        currentRow = self.shaderList.currentRow()
        if currentRow < 0:
            self.refreshShaderProperties()
            return

        shader = obj.shaders[currentRow]

        for propertyName in self.shaderPropertyInputs.keys():
            inputField = self.shaderPropertyInputs[propertyName]
            if not hasattr(shader, propertyName):
                continue

            value = inputField.text()
            
            if value == '':
                self.refreshShaderProperties()
                return
            
            if type(inputField) == QLineEdit:
                typeOf = type(getattr(shader, propertyName))
                if typeOf == str:
                    setattr(shader, propertyName, value)
                elif typeOf in [float, int]:
                    setattr(shader, propertyName, float(value))
            elif type(inputField) == QPushButton and inputField.isCheckable():
                setattr(shader, propertyName, inputField.isChecked())
        
        ior = self.iorField.text()

        if ior == '':
            self.refreshShaderProperties()
            return
        
        try:
            obj.material.ior = float(ior)
        except:
            return

        self.sharedState.sceneChanged = True
        self.refreshShaderProperties()
    
    def refreshShaderProperties(self):
        obj = self.scene.getObject(self.sharedState.selectedObject)

        if obj == None:
            for inputField in self.shaderPropertyInputs.values():
                inputField.blockSignals(True)
                if type(inputField) == QLineEdit:
                    inputField.setText('')
                elif type(inputField) == QPushButton and inputField.isCheckable():
                    inputField.setChecked(False)
                elif type(inputField) == QPushButton:
                    inputField.setStyleSheet('QPushButton {{ background-color: #ffffff; color: black; }}')
                inputField.blockSignals(False)
            self.iorField.blockSignals(True)
            self.iorField.setText('')
            self.iorField.blockSignals(False)
            return
        
        currentRow = self.shaderList.currentRow()

        self.shaderList.clear()

        for shader in obj.shaders:
            self.shaderList.addItem(shader.name)

        if currentRow < 0 or currentRow >= len(obj.shaders):
            currentRow = 0
        
        self.shaderList.blockSignals(True)
        self.shaderList.setCurrentRow(currentRow)
        self.shaderList.blockSignals(False)

        shader = obj.shaders[currentRow]

        for propertyName in self.shaderPropertyInputs.keys():
            inputField = self.shaderPropertyInputs[propertyName]
            if not hasattr(shader, propertyName):
                continue

            propertyValue = getattr(shader, propertyName)

            inputField.blockSignals(True)
            if type(inputField) == QLineEdit:
                inputField.setText(str(propertyValue))
            elif type(inputField) == QPushButton and inputField.isCheckable():
                inputField.setChecked(propertyValue)
            elif type(inputField) == QPushButton:
                 inputField.setStyleSheet(f'QPushButton {{ background-color: rgb({propertyValue.r}, {propertyValue.g}, {propertyValue.b}); color: black; }}')
            inputField.blockSignals(False)
        
        self.iorField.blockSignals(True)
        self.iorField.setText(str(obj.material.ior))
        self.iorField.blockSignals(False)
    
    def setupObjectPanel(self):
        self.objectList = QListWidget()
        self.objectList.currentRowChanged.connect(lambda: self.changeSelectedObject(source='list'))
        # self.objectList.currentRowChanged.connect(lambda: print(self.objectList.currentRow()))
        self.objectPanel.panelLayout.addWidget(self.objectList)
        
        self.flipNormalsButton = QPushButton("Flip normals")
        self.flipNormalsButton.clicked.connect(self.flipSelectedNormals)
        self.objectPanel.panelLayout.addWidget(self.flipNormalsButton)
        self.deleteObjectButton = QPushButton("Delete")
        self.deleteObjectButton.clicked.connect(self.deleteObject)
        self.objectPanel.panelLayout.addWidget(self.deleteObjectButton)
        
        self.refreshObjectList()
    
    def refreshObjectList(self):
        self.objectList.clear()
        selectedObj = self.sharedState.selectedObject
        for i, obj in enumerate(self.scene.objects):
            item = QListWidgetItem(obj.name)
            item.setData(0, obj.name)
            # item.setData(0, obj.id)
            item.setData(1, obj.id)
            self.objectList.addItem(item)
            if obj.id == selectedObj:
                self.objectList.blockSignals(True)
                self.objectList.setCurrentRow(i)
                self.objectList.blockSignals(False)
        
    
    def changeSelectedObject(self, id=-1, source='viewport'):
        if source == 'viewport':
            self.sharedState.selectedObject = id
            self.refreshObjectList()
        elif source == 'list':
            if not self.objectList.currentItem():
                return
            self.sharedState.selectedObject = self.objectList.currentItem().data(1)

        self.sharedState.sceneChanged = True
        self.refreshTransformProperties()
        self.refreshShaderProperties()
        self.refreshLightProperties()

    def setupTransformPanel(self):
        ## Setup transform properties region
        self.transformPropertiesRegion = QWidget()
        self.transformPropertiesRegion.setStyleSheet('background-color: #efefef; border-radius: 4px; margin: 4px;')
        self.transformPropertiesRegion.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.transformPropertiesRegionLayout = QVBoxLayout()
        self.transformPropertiesRegionLayout.setAlignment(Qt.AlignTop)
        self.transformPropertiesRegion.setLayout(self.transformPropertiesRegionLayout)
        self.transformPanel.panelLayout.addWidget(self.transformPropertiesRegion)

        ## Setup transform tabs
        self.transformTabs = QTabWidget()
        self.transformTabs.setStyleSheet('margin: 0;')

        self.transformTabsDict = {}

        for transform in ['pos', 'rot', 'scale']:
            tab = QWidget()
            tab.setStyleSheet('margin: 0;')
            tab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
            tabLayout = QVBoxLayout(tab)
            tabLayout.setAlignment(Qt.AlignTop)
            self.transformTabs.addTab(tab, {
                'pos': '↖',
                'rot': '⤾',
                'scale': '↔',
            }[transform])
            self.transformTabsDict[transform] = []
            
            for direction in range(3):
                directionText = ['X', 'Y', 'Z'][direction]

                property = QWidget()
                property.setStyleSheet('margin: 0;')
                propertyLayout = QHBoxLayout()
                property.setLayout(propertyLayout)
                tabLayout.addWidget(property)
                
                dragButton = DragButton(directionText)
                dragButton.setStyleSheet('background-color: #ffffff; padding-left: 2px; padding-right: 2px; margin: 0;')
                ## NOTE # Default values are used because python evaluates them at define time not call time
                dragButton.onDrag.connect(lambda x, y, currentTransform=transform, currentDirection=direction: self.dragTransform((x, y), False, currentTransform, currentDirection))
                dragButton.clicked.connect(lambda currentTransform=transform, currentDirection=direction: self.dragTransform((0, 0), True, currentTransform, currentDirection))
                propertyLayout.addWidget(dragButton)
                
                inputField = QLineEdit(self)
                inputField.setStyleSheet('background-color: #dedede;')
                inputField.setPlaceholderText(directionText)
                ## NOTE # Default values are used because python evaluates them at define time not call time
                inputField.editingFinished.connect(lambda currentTransform=transform, currentDirection=direction, currentField=inputField: self.transformSelected(currentTransform, currentDirection, currentField))
                propertyLayout.addWidget(inputField)
                
                self.transformTabsDict[transform].append(inputField)

        self.transformPropertiesRegionLayout.addWidget(self.transformTabs)
        
        self.setTransformsButton = QPushButton('Set transforms')
        self.setTransformsButton.setStyleSheet('background-color: #ffffff; border-radius: 3px; margin: 0 6px 4px 6px;')
        self.setTransformsButton.clicked.connect(self.setTransforms)
        self.transformPanel.panelLayout.addWidget(self.setTransformsButton)
    
    def setupViewportPanel(self):
        ## Setup viewport tabs
        self.viewportModesTabs = QTabWidget()
        self.viewportModesTabs.setStyleSheet('margin: 0;')
        self.viewportModesTabs.currentChanged.connect(self.setViewportMode)
        self.viewportPanel.panelLayout.addWidget(self.viewportModesTabs)
        
        # region # texel tab
        self.texelTab = QWidget()
        self.texelTab.setStyleSheet('margin: 0;')
        self.texelTab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.texelTabLayout = QVBoxLayout(self.texelTab)
        self.texelTabLayout.setAlignment(Qt.AlignTop)
        self.viewportModesTabs.addTab(self.texelTab, 'O')
        
        # region # debug tab
        self.debugTab = QWidget()
        self.debugTab.setStyleSheet('margin: 0;')
        self.debugTab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.debugTabLayout = QVBoxLayout(self.debugTab)
        self.debugTabLayout.setAlignment(Qt.AlignTop)
        self.viewportModesTabs.addTab(self.debugTab, 'X')
    
    def setupLightPanel(self):
        ## Setup light properties region
        self.lightPropertiesRegion = QWidget()
        self.lightPropertiesRegion.setStyleSheet('background-color: #efefef; border-radius: 4px; margin: 4px;')
        self.lightPropertiesRegion.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.lightPropertiesRegionLayout = QVBoxLayout()
        self.lightPropertiesRegionLayout.setAlignment(Qt.AlignTop)
        self.lightPropertiesRegion.setLayout(self.lightPropertiesRegionLayout)
        self.lightPanel.panelLayout.addWidget(self.lightPropertiesRegion)

        self.lightStrengthField = QLineEdit(self)
        self.lightStrengthField.setStyleSheet('background-color: #dedede;')
        self.lightStrengthField.setPlaceholderText('Strength')
        self.lightStrengthField.editingFinished.connect(self.adjustLightStrength)
        self.lightPropertiesRegionLayout.addWidget(self.lightStrengthField)
    
    def refreshLightProperties(self):
        self.lightStrengthField.blockSignals(True)
        obj = self.scene.getObject(self.sharedState.selectedObject)
        if not obj or obj.type != 'light':
            self.lightStrengthField.setText('')
            return
        self.lightStrengthField.setText(str(obj.strength))
        self.lightStrengthField.blockSignals(False)
    
    def adjustLightStrength(self):
        try:
            strength = float(self.lightStrengthField.text())
            obj = self.scene.getObject(self.sharedState.selectedObject)
            if not obj or obj.type != 'light':
                return
            
            obj.strength = strength
            
            self.sharedState.sceneChanged = True
        except ValueError:
            pass

        self.refreshLightProperties()
    
    def refreshCameraProperties(self):
        self.cameraLengthField.blockSignals(True)
        camera = self.scene.camera
        self.cameraLengthField.setText(str(camera.length))
        self.cameraLengthField.blockSignals(False)
    
    def adjustCamLength(self):
        try:
            length = float(self.cameraLengthField.text())
            camera = self.scene.camera

            camera.length = length
            
            self.sharedState.sceneChanged = True
        except ValueError:
            pass

        self.refreshCameraProperties()
    
    def setupCameraPanel(self):
        ## Setup camera properties region
        self.cameraPropertiesRegion = QWidget()
        self.cameraPropertiesRegion.setStyleSheet('background-color: #efefef; border-radius: 4px; margin: 4px;')
        self.cameraPropertiesRegion.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.cameraPropertiesRegionLayout = QVBoxLayout()
        self.cameraPropertiesRegionLayout.setAlignment(Qt.AlignTop)
        self.cameraPropertiesRegion.setLayout(self.cameraPropertiesRegionLayout)
        self.cameraPanel.panelLayout.addWidget(self.cameraPropertiesRegion)

        self.cameraLengthField = QLineEdit(self)
        self.cameraLengthField.setStyleSheet('background-color: #dedede;')
        self.cameraLengthField.setPlaceholderText('Length')
        self.cameraLengthField.editingFinished.connect(self.adjustCamLength)
        self.cameraPropertiesRegionLayout.addWidget(self.cameraLengthField)

    # MARK: > RENDER PANEL
    ## Setup the render panel
    def setupRenderPanel(self):
        ## Setup options fields
        self.renderOptionsRegion = QWidget()
        self.renderOptionsRegion.setStyleSheet('background-color: #efefef; border-radius: 4px; margin: 4px;')
        self.renderOptionsRegion.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.renderOptionsRegionLayout = QVBoxLayout()
        self.renderOptionsRegionLayout.setAlignment(Qt.AlignTop)
        self.renderOptionsRegion.setLayout(self.renderOptionsRegionLayout)
        self.renderPanel.panelLayout.addWidget(self.renderOptionsRegion)
        self.renderOptionsInputs = {}
        for key in self.renderer.options.keys():
            typeOf = type(self.renderer.options[key])
            if typeOf in [str, float, int]:
                optionInput = QLineEdit(self)
                optionInput.setPlaceholderText(self.camelToWords(key))
                optionInput.setText(str(self.renderer.options[key]))
                optionInput.editingFinished.connect(self.updateRenderOptions)
                optionInput.setStyleSheet('background-color: #dedede; padding-top: 1px; padding-bottom: 1px;')
                optionInput.setStyleSheet('QLineEdit { background-color: #dedede; padding-top: 1px; padding-bottom: 1px; } QLineEdit:hover { background-color: #efdfe1; }')
            elif typeOf == bool:
                optionInput = QPushButton(self.camelToWords(key))
                optionInput.setCheckable(True)
                if self.renderer.options[key]:
                    optionInput.setChecked(True)
                optionInput.clicked.connect(self.updateRenderOptions)
                optionInput.setStyleSheet('QPushButton { background-color: #ffffff; padding-top: 2px; padding-bottom: 2px; } QPushButton:hover { background-color: #ffe1e3; } QPushButton:checked { background-color: #f488b8 }')
            else:
                continue
            
            self.renderOptionsInputs[key] = [optionInput, type(self.renderer.options[key])]
            self.renderOptionsRegionLayout.addWidget(optionInput)
        
        ## Setup render button
        self.renderButton = QPushButton('Render')
        self.renderButton.setStyleSheet('background-color: #ffffff; border-radius: 3px; margin: 0 6px 4px 6px;')
        self.renderButton.clicked.connect(self.renderPhoton)
        self.renderPanel.panelLayout.addWidget(self.renderButton)
    
    def setViewportMode(self, index):
        if index == 0:
            if self.sharedState.engine != 'texel':
                self.sharedState.engine = 'texel'
                self.sharedState.sceneChanged = True
        else:
            if self.sharedState.engine != 'debug':
                self.sharedState.engine = 'debug'
                self.sharedState.sceneChanged = True
        
    # MARK: > UPDATE OPTIONS
    ## Update render options
    def updateRenderOptions(self):
        try:
            for key in self.renderOptionsInputs.keys():
                input = self.renderOptionsInputs[key][0]
                typeOf = self.renderOptionsInputs[key][1]
                
                if typeOf == int:
                    val = int(input.text())
                elif typeOf == float:
                    val = float(input.text())
                elif typeOf == str:
                    val = str(input.text())
                elif typeOf == bool:
                    val = input.isChecked()
                
                self.renderer.options[key] = val

            width = self.renderer.options['width']
            height = self.renderer.options['height']

            ratio = width / height
            
            if ratio <= 1.6:
                scaleFactor = 900 / height
            else:
                scaleFactor = 1440 / width

            self.tracer.options['width'] = int(width*scaleFactor)
            self.tracer.options['height'] = int(height*scaleFactor)
            
            self.sharedState.texelWidth = int(width*scaleFactor)
            self.sharedState.texelHeight = int(height*scaleFactor)

            self.sharedState.sceneChanged = True
        except ValueError:
            self.refreshRenderProperties()
    
    # MARK: > FLIP NORMS
    ## Flip the normals of the selected object
    def flipSelectedNormals(self):
        obj = self.scene.getObject(self.sharedState.selectedObject)
        if obj and obj.type == 'mesh':
            obj.mesh.flipNormals()
        
        self.sharedState.sceneChanged = True
    
    # MARK: > SET TRANSFORMS
    ## Set the transforms of the current selected object
    def setTransforms(self):
        obj = self.scene.getObject(self.sharedState.selectedObject)
        
        if obj != None:
            obj.setTransforms()
        
        self.refreshTransformProperties()
        self.sharedState.sceneChanged = True
    
    # MARK: > REFRESH TRANSFORM
    ## Refresh transform properties
    def refreshTransformProperties(self):
        obj = self.scene.getObject(self.sharedState.selectedObject)
        
        for tab in self.transformTabsDict.keys():
            for i, input in enumerate(self.transformTabsDict[tab]):
                input.blockSignals(True)
                if obj != None:
                    val = {
                        'pos': obj.position,
                        'rot': obj.rotation,
                        'scale': obj.scale,
                    }[tab][i]
                    
                    print(val)
                    if tab == 'rot':
                        val = math.degrees(val)
                    print(val)

                    input.setText(f'{val:.3f}')
                else:
                    input.setText('')
                input.blockSignals(False)
    
    # MARK: > REFRESH RENDER
    ## Refresh render properties
    def refreshRenderProperties(self):
        for key in self.renderOptionsInputs.keys():
            optionInput = self.renderOptionsInputs[key][0]
            typeOf = self.renderOptionsInputs[key][1]
            val = self.renderer.options[key]
            
            optionInput.blockSignals(True)

            if typeOf in [str, float, int]:
                optionInput.setText(str(val))
            elif typeOf == bool:
                if val:
                    optionInput.setChecked(True)
            
            optionInput.blockSignals(False)
    
    def dragTransform(self, delta, first, transform, direction):
        obj = self.scene.getObject(self.sharedState.selectedObject)
        if obj != None:
            if first:
                print('setting')
                self.originalTransforms = [copy.deepcopy(obj.position), copy.deepcopy(obj.rotation), copy.deepcopy(obj.scale)]
            deltaX = delta[0]
            if transform == 'pos':
                print('before:', self.originalTransforms[0])
                scaled = deltaX / 60
                transforms = copy.deepcopy(self.originalTransforms[0])
                # print(obj.position)
                # print(scaled)
                obj.position[direction] = transforms[direction] + scaled
                # print(obj.position)
                print('after:', self.originalTransforms[0])
            if transform == 'rot':
                scaled = deltaX / 2
                obj.rotation[direction] = math.radians(math.degrees(self.originalTransforms[1][direction]) + scaled)
            if transform == 'scale':
                scaled = (deltaX / 60)
                if scaled == -self.originalTransforms[2][direction]: scaled += 0.0001
                obj.scale[direction] = self.originalTransforms[2][direction] + scaled
            
            self.sharedState.sceneChanged = True
            self.refreshTransformProperties()
    
    # MARK: > TRANSFORM SEL
    ## Transform selected object
    def transformSelected(self, transform, direction, field):
        obj = self.scene.getObject(self.sharedState.selectedObject)
        
        if obj != None:
            try:
                if transform == 'pos':
                    obj.position[direction] = float(field.text())
                elif transform == 'rot':
                    obj.rotation[direction] = math.radians(float(field.text()))
                elif transform == 'scale':
                    obj.scale[direction] = float(field.text())
            except ValueError:
                self.refreshTransformProperties()
                QMessageBox.warning(self, 'Error', 'Please enter valid numbers in all fields.')
            
        self.refreshTransformProperties()
        self.sharedState.sceneChanged = True

    def cancelRender(self):
        self.sharedState.renderCancelled = True
        self.progressDialog.close()

    # MARK: > RENDER
    ## Render scene
    def renderPhoton(self):
        self.sharedState.renderCancelled = False

        self.progressDialog = QProgressDialog('Rendering...', 'Cancel', 0, 100, self)
        self.progressDialog.setWindowModality(Qt.ApplicationModal)
        self.progressDialog.setWindowTitle('Render')
        self.progressDialog.setMinimumDuration(0)
        self.progressDialog.setValue(0)
        self.progressDialog.canceled.connect(self.cancelRender)

        self.renderThread = RenderThread(photon=self.renderer, sharedState=self.sharedState)
        self.renderThread.progressUpdate.connect(self.updateRenderProgress)
        self.renderThread.renderComplete.connect(self.completeRender)
        
        self.renderThread.start()
    
    def completeRender(self, data):
        self.progressDialog.close()
        self.displayRenderedImage(data)
        self.lastRender = data
    
    # MARK: > PROGRESS
    ## Update progress bars to show render progress
    def updateRenderProgress(self, data, key, total):
        avgProg = sum([value/total for value in data.values()])
            
        self.progressDialog.setValue(int(avgProg * 100))
            
        if all([val/total == 1 for val in data.values()]):
            self.progressDialog.setLabelText('Merging ...')
        
        # print(f'Thread {key}: {data[key]/total}%')

    
    # MARK: > VPORT CLICK
    ## Handle when the viewport is clicked
    @pyqtSlot(int, int)
    def handleViewportClick(self, x, y):
        if self.sharedState.engine != 'texel':
            return
        
        ## Fire a ray through the pixel
        pixelVec = self.tracer.pixelToVector(x, y)
        ray = self.tracer.getRay(pixelVec)
        
        ## Get the first object that the ray collides with
        collisionInfo = self.tracer.getFirstCollision(ray, lights=True, backCull=False)
        
        object = collisionInfo['object']
        
        if self.facesMode:
            if object == None or object.id != self.sharedState.selectedObject:
                if not self.shiftHeld:
                    self.sharedState.selectedFaces = []
            else:
                faceIndex = object.mesh.faces.index(collisionInfo['face'])
            
                if self.shiftHeld:
                    if faceIndex in self.sharedState.selectedFaces:
                        self.sharedState.selectedFaces.remove(faceIndex)
                    else:
                        self.sharedState.selectedFaces.append(faceIndex)
                else:
                    if faceIndex in self.sharedState.selectedFaces:
                        if len(self.sharedState.selectedFaces) > 1:
                            self.sharedState.selectedFaces = [faceIndex]
                        else:
                            self.sharedState.selectedFaces = []
                    else:
                        self.sharedState.selectedFaces = [faceIndex]
                        

        else:
            ## Compare original selected object and new object
            originalObj = self.sharedState.selectedObject
            newObj = object.id if object else -1
            
            ## Only change if they are not the same
            if originalObj != newObj:
                if object == None:
                    ## Ray did not hit anything
                    self.changeSelectedObject(-1, 'viewport')
                else:
                    self.changeSelectedObject(object.id, 'viewport')
        
        self.sharedState.sceneChanged = True
        
    
    # MARK: > DISPLAY IMG
    ## Convert the image data into a pixmap and display it
    # @pyqtSlot(np.ndarray)
    def displayRenderedImage(self, imageData):
        # Get dimensions
        height, width, _ = imageData.shape
        
        self.viewport.setFixedSize(width, height)

        ## Convert the NumPy array to a QImage
        # qImage = QImage(imageData.data, width, height, imageData.strides[0], QImage.Format_RGB888)
        qImage = qimage2ndarray.array2qimage(imageData)

        ## Set the pixmap as the image label
        pixmap = QPixmap.fromImage(qImage)
        self.viewport.setPixmap(pixmap)

    # MARK: > TOGGLE NORMALS
    ## Toggle visibility of normal direction in viewport
    def toggleNormals(self):
        ## Update normals visiblity
        self.sharedState.normals = not self.sharedState.normals
        
        ## Trigger scene render
        self.sharedState.sceneChanged = True


# MARK: MAIN
## Main application 
def main():
    ## Setup app and window
    app = QApplication(sys.argv)
    window = MainWindow()
    
    ## Show window
    window.show()
    
    ## Exit when app is closed
    sys.exit(app.exec_())


## Only run when program is run as main
if __name__ == '__main__':
    main()