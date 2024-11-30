# AQA A-Level Computer Science NEA 2024
#
# Graphics Engine
# Theo HT
#
# (Explorer)
#
# NOTE # Comments with `MARK` in them are purely IDE-related. They have no relevance to the code.

# MARK: IMPORTS
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QFileDialog, QLineEdit, QTabWidget, QSizePolicy, QProgressDialog, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt, QCoreApplication
from T3DE import Scene, RGB, Vec3, Euler, Scale
from PyQt5.QtGui import QImage, QPixmap
from T3DR import Texel, Photon, Debug
from string import ascii_uppercase
from functools import partial
from threading import Lock
import numpy as np
import math
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
    
    # MARK: > SCENE CHANGED
    @property
    def sceneChanged(self):
        with self.lock:
            return self.__sceneChanged

    @sceneChanged.setter
    def sceneChanged(self, sceneChanged):
        with self.lock:
            self.__sceneChanged = sceneChanged
    
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
        
        self.panel.setStyleSheet('background-color: #e3e3e3; border-top-left-radius: 0; border-top-right-radius: 0; border-bottom-left-radius: 5px; border-bottom-right-radius: 5px;')
        self.toggle.setStyleSheet('')

        self.expanded = False

        # self.panel.setFixedHeight(self.panel.sizeHint().height())
        self.panel.setFixedHeight(0)

    # MARK: > TOGGLE
    ## Toggle panel visiblity
    def togglePanel(self):
        # if self.toggle.isChecked():
        if not self.expanded:
            self.toggle.setStyleSheet('background-color: #e3e3e3; border-top-left-radius: 5px; border-top-right-radius: 5px; border-bottom-left-radius: 0; border-bottom-right-radius: 0; padding-top: 3px; padding-bottom: 1px;')
            self.toggle.setText('V ' + self.title)
            self.expanded = True
            self.panel.setFixedHeight(self.panel.sizeHint().height())
        else:
            self.toggle.setStyleSheet('')
            self.toggle.setText('> ' + self.title)
            self.expanded = False
            self.panel.setFixedHeight(0)


# MARK: VIEWPORT THREAD
## Separate thread for rendering viewport concurrently with the GUI
class ViewportThread(QThread):
    ## Setup render complete signal
    renderComplete = pyqtSignal(np.ndarray)
    
    def __init__(self, scene, sharedState, parent=None):
        super().__init__(parent)
        self.sharedState = sharedState
        self.texel = Texel(scene=scene, options={
            'edges': True,
            'ambient': RGB(32, 32, 32),
            'selectedObject': self.sharedState.selectedObject,
            'normals': self.sharedState.normals,
        })
        self.debug = Debug(scene=scene, options={
            'direction': 'side',
            'size': 3
        })

    # MARK: > RUN
    ## Render the scene continuously
    def run(self):
        while True:
            ## Only run if scene has changed
            if self.sharedState.sceneChanged:
                ## Change the selected object gizmo
                self.texel.options['selectedObject'] = self.sharedState.selectedObject
                self.texel.options['normals'] = self.sharedState.normals

                engine = self.texel if self.sharedState.engine == 'texel' else self.debug
                
                ## Get image data from render
                render = engine.render()
                renderedData = render.imageAs('np')

                ## Send the signal back to the connected slot
                self.renderComplete.emit(renderedData)
                
                self.sharedState.sceneChanged = False

                # ## Add delay between renders to balance performance
                # self.msleep(1000)


# MARK: RENDER THREAD
## Separate thread for rendering final render
class RenderThread(QThread):
    ## Setup progress update signal
    progressUpdate = pyqtSignal(dict, int)
    ## Setup render complete signal
    renderComplete = pyqtSignal(np.ndarray)
    
    def __init__(self, photon, sharedState, parent=None):
        super().__init__(parent)
        self.photon = photon
        self.sharedState = sharedState
    
    def run(self):
        # render = self.photon.render(progressCallback=self.progressUpdate.emit)
        render = self.photon.render(progressCallback=lambda data, key: self.progressUpdate.emit(data, key), cancelCallback=lambda: self.sharedState.renderCancelled)
        renderedData = render.imageAs('np')
        
        self.renderComplete.emit(renderedData)


# MARK: MAIN WINDOW
## Main application window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        ## Setup shared state for global information
        self.sharedState = SharedState()
        
        ## Setup scene for explorer
        filepath = QFileDialog.getOpenFileName()[0]
        self.scene = Scene.fromJSON(filepath)
        
        ## Build BVH for all objects
        for obj in self.scene.objects:
            obj.bvh = obj.buildBVH()

        ## Setup photon instance for ray-tracing to detect clicks
        self.tracer = Photon(scene=self.scene, options={
            'aabb': True,
            'bvh': True,
            'debug': False,
            'transformBefore': False,
        })
        
        ## Setup photon instance for the actual render
        self.renderer = Photon(scene=self.scene, options={
            'progressMode': 'none',
            
            'step': 80,
            'fillStep': True,
            
            'bounces': 2,
            'samples': 3,
            
            'aabb': True,
            'bvh': True,
            
            'wait': False,
            'debug': False,
            
            'lights': True,
            'emissionSampleDensity': 1,
            
            'ambient': RGB(32, 32, 32),
            
            'threads': 2,
        })

        ## Setup window
        self.setWindowTitle('3D Explorer')
        self.setGeometry(100, 100, 800, 600)

        ## Setup central widget
        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.layout = QHBoxLayout()
        self.centralWidget.setLayout(self.layout)

        ## Setup viewport
        self.viewport = ClickableImage('...')
        self.viewport.clicked.connect(self.handleViewportClick)
        self.viewport.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.viewport)

        ## Setup controls layout
        self.controlsLayout = QVBoxLayout()
        self.controlsLayout.setAlignment(Qt.AlignTop)
        self.layout.addLayout(self.controlsLayout)

        ## Setup button to toggle normals visibility
        self.normalsButton = QPushButton('Normals')
        self.normalsButton.setCheckable(True)
        self.normalsButton.clicked.connect(self.toggleNormals)
        self.controlsLayout.addWidget(self.normalsButton)

        ## Setup button to flip normals of selected object
        self.flipNormalsButton = QPushButton('Flip normals')
        self.flipNormalsButton.clicked.connect(self.flipSelectedNormals)
        self.controlsLayout.addWidget(self.flipNormalsButton)

        ## Setup render panel
        self.renderPanel = CollapsiblePanel('Render')
        self.renderPanel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.controlsLayout.addWidget(self.renderPanel)
        self.setupRenderPanel()

        ## Setup render panel
        self.viewportPanel = CollapsiblePanel('Viewport')
        self.viewportPanel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.controlsLayout.addWidget(self.viewportPanel)
        self.setupViewportPanel()

        ## Setup transform panel
        self.transformPanel = CollapsiblePanel('Transform')
        self.transformPanel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.controlsLayout.addWidget(self.transformPanel)
        self.setupTransformPanel()

        ## Setup threading for separate viewport render
        self.viewportThread = ViewportThread(scene=self.scene, sharedState=self.sharedState)
        self.viewportThread.renderComplete.connect(self.displayRenderedImage)

        self.refreshTransformProperties()
        self.refreshRenderProperties()
        self.viewportThread.start()
    
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
        
        # region # position tab
        self.positionTab = QWidget()
        self.positionTab.setStyleSheet('margin: 0;')
        self.positionTab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.positionTabLayout = QVBoxLayout(self.positionTab)
        self.positionTabLayout.setAlignment(Qt.AlignTop)
        self.transformTabs.addTab(self.positionTab, '↖')

        self.positionXInput = QLineEdit(self)
        self.positionXInput.setStyleSheet('background-color: #dedede;')
        self.positionXInput.setPlaceholderText('X')
        self.positionXInput.editingFinished.connect(lambda: self.transformSelected('pos'))
        self.positionTabLayout.addWidget(self.positionXInput)
        
        self.positionYInput = QLineEdit(self)
        self.positionYInput.setStyleSheet('background-color: #dedede;')
        self.positionYInput.setPlaceholderText('Y')
        self.positionYInput.editingFinished.connect(lambda: self.transformSelected('pos'))
        self.positionTabLayout.addWidget(self.positionYInput)

        self.positionZInput = QLineEdit(self)
        self.positionZInput.setStyleSheet('background-color: #dedede;')
        self.positionZInput.setPlaceholderText('Z')
        self.positionZInput.editingFinished.connect(lambda: self.transformSelected('pos'))
        self.positionTabLayout.addWidget(self.positionZInput)
        # endregion
        
        # region # rotation tab
        self.rotationTab = QWidget()
        self.rotationTab.setStyleSheet('margin: 0;')
        self.rotationTab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.rotationTabLayout = QVBoxLayout(self.rotationTab)
        self.rotationTabLayout.setAlignment(Qt.AlignTop)
        self.transformTabs.addTab(self.rotationTab, '⤾')

        self.rotationXInput = QLineEdit(self)
        self.rotationXInput.setStyleSheet('background-color: #dedede;')
        self.rotationXInput.setPlaceholderText('X')
        self.rotationXInput.editingFinished.connect(lambda: self.transformSelected('rot'))
        self.rotationTabLayout.addWidget(self.rotationXInput)
        
        self.rotationYInput = QLineEdit(self)
        self.rotationYInput.setStyleSheet('background-color: #dedede;')
        self.rotationYInput.setPlaceholderText('Y')
        self.rotationYInput.editingFinished.connect(lambda: self.transformSelected('rot'))
        self.rotationTabLayout.addWidget(self.rotationYInput)

        self.rotationZInput = QLineEdit(self)
        self.rotationZInput.setStyleSheet('background-color: #dedede;')
        self.rotationZInput.setPlaceholderText('Z')
        self.rotationZInput.editingFinished.connect(lambda: self.transformSelected('rot'))
        self.rotationTabLayout.addWidget(self.rotationZInput)
        # endregion
        
        # region # scale tab
        self.scaleTab = QWidget()
        self.scaleTab.setStyleSheet('margin: 0;')
        self.scaleTab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.scaleTabLayout = QVBoxLayout(self.scaleTab)
        self.scaleTabLayout.setAlignment(Qt.AlignTop)
        self.transformTabs.addTab(self.scaleTab, '↔')

        self.scaleXInput = QLineEdit(self)
        self.scaleXInput.setStyleSheet('background-color: #dedede;')
        self.scaleXInput.setPlaceholderText('X')
        self.scaleXInput.editingFinished.connect(lambda: self.transformSelected('scale'))
        self.scaleTabLayout.addWidget(self.scaleXInput)
        
        self.scaleYInput = QLineEdit(self)
        self.scaleYInput.setStyleSheet('background-color: #dedede;')
        self.scaleYInput.setPlaceholderText('Y')
        self.scaleYInput.editingFinished.connect(lambda: self.transformSelected('scale'))
        self.scaleTabLayout.addWidget(self.scaleYInput)

        self.scaleZInput = QLineEdit(self)
        self.scaleZInput.setStyleSheet('background-color: #dedede;')
        self.scaleZInput.setPlaceholderText('Z')
        self.scaleZInput.editingFinished.connect(lambda: self.transformSelected('scale'))
        self.scaleTabLayout.addWidget(self.scaleZInput)
        # endregion
        
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
        self.viewportModesTabs.addTab(self.texelTab, 'O')
        
        # region # debug tab
        self.debugTab = QWidget()
        self.debugTab.setStyleSheet('margin: 0;')
        self.debugTab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.viewportModesTabs.addTab(self.debugTab, 'X')
    
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
            elif typeOf == bool:
                optionInput = QPushButton(self.camelToWords(key))
                optionInput.setCheckable(True)
                if self.renderer.options[key]:
                    optionInput.setChecked(True)
                optionInput.clicked.connect(self.updateRenderOptions)
                optionInput.setStyleSheet('QPushButton { background-color: #ffffff; padding-top: 2px; padding-bottom: 2px; } QPushButton:hover { background-color: #FFE1E3; } QPushButton:checked { background-color: #F488B8 }')
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
        
        self.positionXInput.blockSignals(True)
        self.positionYInput.blockSignals(True)
        self.positionZInput.blockSignals(True)
        self.rotationXInput.blockSignals(True)
        self.rotationYInput.blockSignals(True)
        self.rotationZInput.blockSignals(True)
        self.scaleXInput.blockSignals(True)
        self.scaleYInput.blockSignals(True)
        self.scaleZInput.blockSignals(True)
        if obj != None:
            self.positionXInput.setText(str(obj.position.x))
            self.positionYInput.setText(str(obj.position.y))
            self.positionZInput.setText(str(obj.position.z))
            self.rotationXInput.setText(str(math.degrees(obj.rotation.x)))
            self.rotationYInput.setText(str(math.degrees(obj.rotation.y)))
            self.rotationZInput.setText(str(math.degrees(obj.rotation.z)))
            self.scaleXInput.setText(str(obj.scale.x))
            self.scaleYInput.setText(str(obj.scale.y))
            self.scaleZInput.setText(str(obj.scale.z))
        else:
            self.positionXInput.setText('')
            self.positionYInput.setText('')
            self.positionZInput.setText('')
            self.rotationXInput.setText('')
            self.rotationYInput.setText('')
            self.rotationZInput.setText('')
            self.scaleXInput.setText('')
            self.scaleYInput.setText('')
            self.scaleZInput.setText('')
        self.positionXInput.blockSignals(False)
        self.positionYInput.blockSignals(False)
        self.positionZInput.blockSignals(False)
        self.rotationXInput.blockSignals(False)
        self.rotationYInput.blockSignals(False)
        self.rotationZInput.blockSignals(False)
        self.scaleXInput.blockSignals(False)
        self.scaleYInput.blockSignals(False)
        self.scaleZInput.blockSignals(False)
    
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
    
    # MARK: > TRANSFORM SEL
    ## Transform selected object
    def transformSelected(self, transform):
        try:
            ## Get selected object
            obj = self.scene.getObject(self.sharedState.selectedObject)
            
            if obj != None:
                originalPos = obj.position
                originalRot = obj.rotation
                originalScale = obj.scale
                if transform == 'pos':
                    ## Get inputs
                    x = float(self.positionXInput.text())
                    y = float(self.positionYInput.text())
                    z = float(self.positionZInput.text())
                    
                    obj.position = Vec3(x, y, z)
                elif transform == 'rot':
                    ## Get inputs
                    x = float(self.rotationXInput.text())
                    y = float(self.rotationYInput.text())
                    z = float(self.rotationZInput.text())
            
                    obj.rotation = Euler(math.radians(x), math.radians(y), math.radians(z))
                elif transform == 'scale':
                    ## Get inputs
                    x = float(self.scaleXInput.text())
                    y = float(self.scaleYInput.text())
                    z = float(self.scaleZInput.text())

                    obj.scale = Scale(x, y, z)
                
                if originalPos != obj.position or originalRot != obj.rotation or originalScale != obj.scale:
                    ## Trigger refresh
                    self.refreshTransformProperties()
                    self.sharedState.sceneChanged = True
            else:
                self.refreshTransformProperties()
        except ValueError:
            self.refreshTransformProperties()
            QMessageBox.warning(self, 'Error', 'Please enter valid numbers in all fields.')

    def cancelRender(self):
        self.sharedState.renderCancelled = True

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
    
    # MARK: > PROGRESS
    ## Update progress bars to show render progress
    def updateRenderProgress(self, data, key):
        avgProg = sum([value[0]/value[1] for value in data.values()])
            
        self.progressDialog.setValue(int(avgProg * 100))
            
        if all([val[0]/val[1] >= 0.99 for val in data.values()]):
            self.progressDialog.setLabelText('Merging ...')
        
    
    # MARK: > VPORT CLICK
    ## Handle when the viewport is clicked
    @pyqtSlot(int, int)
    def handleViewportClick(self, x, y):
        if self.sharedState.engine != 'texel':
            return
        ## Fire a ray through that pixel
        pixelVec = self.tracer.pixelToVector(x, y)
        ray = self.tracer.getRay(pixelVec)
        
        ## Get the first object that the ray collides with
        collisionInfo = self.tracer.getCollision(ray, lights=False, backCull=True)
        object = collisionInfo['object']
        
        ## Compare original selected object and new object
        originalObj = self.sharedState.selectedObject
        newObj = object.id if object else -1
        
        ## Only change if they are not the same
        if originalObj != newObj:
            if object == None:
                ## Ray did not hit anything
                self.sharedState.selectedObject = -1
            else:
                self.sharedState.selectedObject = object.id
            
            ## Trigger scene render
            self.sharedState.sceneChanged = True
        
        self.refreshTransformProperties()
        
    
    # MARK: > DISPLAY IMG
    ## Convert the image data into a pixmap and display it
    # @pyqtSlot(np.ndarray)
    def displayRenderedImage(self, imageData):
        # Get dimensions
        height, width, _ = imageData.shape

        ## Convert the NumPy array to a QImage
        qImage = QImage(imageData.data, width, height, imageData.strides[0], QImage.Format_RGB888)

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