# AQA A-Level Computer Science NEA 2024
#
# Graphics Engine
# Theo HT
#
# (Rendering Engines)
#
# NOTE # Comments with `MARK` in them are purely IDE-related. They have no relevance to the code.

# MARK: IMPORTS
from T3DE import Vec3, Ray, MathematicalPlane, MathematicalSphere, RGB, TraceTreeNode, Scale, Mesh, Euler
from alive_progress import alive_bar
from copy import deepcopy
from queue import Queue
import numpy as np
import threading
import random
import math
import time
import copy
import json
import cv2

# MARK: NBC ERROR
## Custom error for when no camera is bound to the scene
class NoBoundCameraError(Exception):
    def __init__(self, message="Scene contains no bound camera. Bind a camera in order to render.\n\n\t\tScene().camera = Camera()\n\n\tOR\tcamera = Camera(Scene())\n\n"):
        self.message = message
        super().__init__(self.message)


# MARK: RENDERED
## Essentially an image, but with extra information about how it was rendered
class Rendered:
    def __init__(self, imgData, traces, width, height, time=None, rays=None, options=None, failed=False, engine=None):
        self.imgData = np.flip(imgData, 2) ## Convert BGR to RGB
        self.traces = traces
        self.width = width
        self.height = height
        self.time = time
        self.rays = rays
        self.options = options
        self.failed = failed
        self.engine = engine
    
    ## Return the image in different forms
    def imageAs(self, type):
        match type:
            case 'list':
                return self.imgData.toList()
            case 'cv2': ## Convert RGB to BGR
                return np.flip(self.imgData, 2)
            case 'np':
                return self.imgData
    
    def toDict(self):
        options = self.options
        if 'ambient' in options.keys():
            options['ambient'] = [*options['ambient']]
        return {
            'imgData': [[
                [int(pix[0]), int(pix[1]), int(pix[2])]
                for pix in row]
                for row in self.imgData
            ],
            'traces': {
                str(pixel[0]) + ',' + str(pixel[1]): [trace.toDict() if trace != None else None for trace in pixTraces] 
                for pixel, pixTraces in self.traces.items()
            },
            'width': self.width,
            'height': self.height,
            'time': self.time,
            'rays': self.rays,
            'options': options,
            'failed': self.failed,
            'engine': self.engine,
        }
    
    @staticmethod
    def fromDict(data):
        options = data.get('options', {})
        
        if 'ambient' in options.keys():
            options['ambient'] = RGB(*options['ambient'])
        return Rendered(
            imgData=[[
                (int(pix[2]), int(pix[1]), int(pix[0]))
                for pix in row]
                for row in data.get('imgData', [])
            ],
            traces={
                (int(pixel.split(',')[0]), int(pixel.split(',')[1])): [TraceTreeNode.fromDict(trace) if trace != None else None for trace in pixTraces] 
                for pixel, pixTraces in data.get('traces', {}).items()
            },
            width=data.get('width', 1440),
            height=data.get('height', 900),
            time=data.get('time', 0),
            rays=data.get('rays', 0),
            options=options,
            failed=data.get('failed', False),
            engine=data.get('engine', 'debug'),
        )
    
    def saveJSON(self, filepath):
        data = self.toDict()
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    @staticmethod
    def fromJSON(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return Rendered.fromDict(data)


# MARK: DEBUG
## Provide an orthographic line-drawing of the scene for debug purposes
class Debug:
    def __init__(self, scene, options={}) -> None:
        self.scene = scene
        self.transformedScene = self.scene
        self.options = self.mergeDefaultOptions(options)
    
    # MARK: > ADMIN
    ## Make sure all options exist even when unspecified
    def mergeDefaultOptions(self, options):
        defaultOptions = {'size':1, 'width':1440, 'height':900, 'direction': 'top', 'lights': True, 'camera': True, 'aabb': True, 'bvh': False, 'debug': True}
        for optionKey in defaultOptions.keys():
            if optionKey not in options.keys():
                options[optionKey] = defaultOptions[optionKey]
        
        options['size'] /= 100
        
        return options

    ## Size needs to be s much smaller number than is practical
    def setSize(self, size):
        self.options['size'] = size/100
    
    ## Get the necessary coordinate components depending on direction of view
    def mapCoordinates(self, x, y, z):
        direction = self.options['direction']
        if direction == 'top':
            return -x, -z
        elif direction == 'side':
            return z, -y
        elif direction == 'front':
            return x, -y
        else:
            raise ValueError(f"Invalid direction '{direction}'")
    
    # MARK: > TRACE
    ## (Recursive) Display a ray trace
    def renderTraceNode(self, image, traceNode, halfWidth, halfHeight, color):
        ## Base case
        if traceNode is None:
            return

        ray = traceNode.ray

        ## Get the ray start position
        rayVectorX, rayVectorY = self.mapCoordinates(*ray.start)
        rayStartPixelX = round((rayVectorX / self.options['size']) + halfWidth)
        rayStartPixelY = round((rayVectorY / self.options['size']) + halfHeight)
        
        if None not in traceNode.collisionInfo:
            collisionPoint = traceNode.collisionInfo['coordinate']
            # print(collisionPoint)

            ## Get the ray end position
            collisionPointVectorX, collisionPointVectorY = self.mapCoordinates(*collisionPoint)
            rayEndPixelX = round((collisionPointVectorX / self.options['size']) + halfWidth)
            rayEndPixelY = round((collisionPointVectorY / self.options['size']) + halfHeight)
            
            ## Draw ray
            cv2.line(image, (rayStartPixelX, rayStartPixelY), (rayEndPixelX, rayEndPixelY), color, thickness=1)
            
            collisionColor = (0, 0, 255) if traceNode.collisionInfo['direction'] == 0 else (0, 255, 0)
            
            ## Draw collision
            cv2.circle(image, (rayEndPixelX, rayEndPixelY), 2, collisionColor, thickness=-1)
            
            normalEnd = collisionPoint + traceNode.collisionInfo['normal']
            normalEndVectorX, normalEndVectorY = self.mapCoordinates(*normalEnd)
            normalEndPixelX = round((normalEndVectorX / self.options['size']) + halfWidth)
            normalEndPixelY = round((normalEndVectorY / self.options['size']) + halfHeight)
            
            ## Draw normal
            # cv2.line(image, (rayEndPixelX, rayEndPixelY), (0, 0), (255, 133, 255), thickness=1)
            cv2.line(image, (rayEndPixelX, rayEndPixelY), (normalEndPixelX, normalEndPixelY), (255, 133, 255), thickness=1)
        else:
            ## Get an arbitrary 'infinite' (finite in reality) ray end position
            rayEndVectorX, rayEndVectorY = self.mapCoordinates(*(ray.start + (ray.direction*100)))
            rayEndPixelX = round((rayEndVectorX / self.options['size']) + halfWidth)
            rayEndPixelY = round((rayEndVectorY / self.options['size']) + halfHeight)
            
            # print(rayStartPixelX, rayStartPixelY, rayEndPixelX, rayEndPixelY)
            # print(type(rayStartPixelX), type(rayStartPixelY), type(rayEndPixelX), type(rayEndPixelY))
            
            ## Draw ray
            cv2.line(image, (rayStartPixelX, rayStartPixelY), (rayEndPixelX, rayEndPixelY), color, thickness=1)

        ## Recursion
        if traceNode.left is not None:
            self.renderTraceNode(image, traceNode.left, halfWidth, halfHeight, (0, 80, 200))
        if traceNode.right is not None:
            self.renderTraceNode(image, traceNode.right, halfWidth, halfHeight, (200, 80, 0))

    # MARK: > CAMERA
    ## Draw the camera
    def renderCamera(self, image, halfWidth, halfHeight):
        camera = self.transformedScene.camera
        
        ## Get camera position
        cameraVectorX, cameraVectorY = self.mapCoordinates(*camera.position)
        cameraPixelX = round((cameraVectorX / self.options['size']) + halfWidth)
        cameraPixelY = round((cameraVectorY / self.options['size']) + halfHeight)
        
        ## Draw camera focal point
        # cv2.circle(image, (cameraPixelX, cameraPixelY), 2, camera.shader.debugColor.toTuple('bgr'), thickness=-1)
        cv2.circle(image, (cameraPixelX, cameraPixelY), 2, camera.shaders[0].debugColor.toTuple('bgr'), thickness=-1)

        ## Get image plane vertices
        vertices = [
            (Vec3(0, 0, camera.length/20) - Vec3((self.options['width']/self.options['height']) / 2, 0, 0)).applyTransforms(camera.position, camera.scale, camera.rotation),
            (Vec3(0, 0, camera.length/20) + Vec3((self.options['width']/self.options['height']) / 2, 0, 0)).applyTransforms(camera.position, camera.scale, camera.rotation),
            (Vec3(0, 0, camera.length/20) - Vec3(0, 0.5, 0)).applyTransforms(camera.position, camera.scale, camera.rotation),
            (Vec3(0, 0, camera.length/20) + Vec3(0, 0.5, 0)).applyTransforms(camera.position, camera.scale, camera.rotation),
        ]

        for vertex in vertices:
            ## Get vertex position
            pixelVectorX, pixelVectorY = self.mapCoordinates(*vertex)
            pixelX = round((pixelVectorX / self.options['size']) + halfWidth)
            pixelY = round((pixelVectorY / self.options['size']) + halfHeight)

            for otherVertex in vertices:
                ## Get other vertex position
                otherPixelVectorX, otherPixelVectorY = self.mapCoordinates(*otherVertex)
                otherPixelX = round((otherPixelVectorX / self.options['size']) + halfWidth)
                otherPixelY = round((otherPixelVectorY / self.options['size']) + halfHeight)

                ## Draw line between both vertices
                cv2.line(
                    image,
                    (otherPixelX, otherPixelY),
                    (pixelX, pixelY),
                    # camera.shader.debugColor.toTuple('bgr'),
                    camera.shaders[0].debugColor.toTuple('bgr'),
                    thickness=1
                )

            ## Draw line from vertex to focal point
            # cv2.line(image, (pixelX, pixelY), (cameraPixelX, cameraPixelY), camera.shader.debugColor.toTuple('bgr'), thickness=1)
            cv2.line(image, (pixelX, pixelY), (cameraPixelX, cameraPixelY), camera.shaders[0].debugColor.toTuple('bgr'), thickness=1)

            directionX = pixelX - cameraPixelX
            directionY = pixelY - cameraPixelY

            extendedPixelX = pixelX + int(directionX / (directionX**2 + directionY**2)**0.5 * 400)
            extendedPixelY = pixelY + int(directionY / (directionX**2 + directionY**2)**0.5 * 400)

            ## Draw line outwards to show field of view
            cv2.line(image, (pixelX, pixelY), (extendedPixelX, extendedPixelY), (128, 128, 128), thickness=1)

    # MARK: > BVH
    ## (Recursive) Draw BVH
    def renderBVH(self, image, halfWidth, halfHeight, bvh, maxDepth, depth=0):
        ## Base cases
        if bvh is None or bvh.box == None or depth == maxDepth:
            return

        boxMin = bvh.box[0]
        boxMax = bvh.box[1]
        # print(boxMin, boxMax)

        ## Get pixels
        pixelMinVectorX, pixelMinVectorY = self.mapCoordinates(*boxMin)
        pixelMinX = round((pixelMinVectorX / self.options['size']) + halfWidth)
        pixelMinY = round((pixelMinVectorY / self.options['size']) + halfHeight)

        pixelMaxVectorX, pixelMaxVectorY = self.mapCoordinates(*boxMax)
        pixelMaxX = round((pixelMaxVectorX / self.options['size']) + halfWidth)
        pixelMaxY = round((pixelMaxVectorY / self.options['size']) + halfHeight)

        ## Draw rectangle
        cv2.rectangle(image, (pixelMinX, pixelMinY), (pixelMaxX, pixelMaxY), (200, 100, 200), 1)

        ## Recursion
        self.renderBVH(image, halfWidth, halfHeight, bvh.left, 20)
        self.renderBVH(image, halfWidth, halfHeight, bvh.right, 20)

    # MARK: > AXES
    ## Draw the axes with markers on the orthographic view
    def renderAxes(self, image, halfWidth, halfHeight):
        ## Define the midpoint of each of the window edges
        topMidPoint = (int(halfWidth), 0)
        bottomMidPoint = (int(halfWidth), int(self.options['height']))
        leftMidPoint = (0, int(halfHeight))
        rightMidPoint = (int(self.options['width']), int(halfHeight))
        
        ## Draw axes
        cv2.line(image, topMidPoint, bottomMidPoint, (0, 0, 0), 1)
        cv2.line(image, leftMidPoint, rightMidPoint, (0, 0, 0), 1)
        
        ## Draw vertical axis marker y values
        for y in range(-40, 40):
            pixelY = round((y / self.options['size']) + int(halfHeight))
            if pixelY < 0 or pixelY > self.options['height']:
                continue
            
            leftPoint = (int(halfWidth-5), pixelY)
            rightPoint = (int(halfWidth+5), pixelY)
            cv2.line(image, leftPoint, rightPoint, (0, 0, 0), 1)
        
        ## Draw horizontal axis marker x values
        for x in range(-40, 40):
            pixelX = round((x / self.options['size']) + int(halfWidth))
            if pixelX < 0 or pixelX > self.options['width']:
                continue
            
            leftPoint = (pixelX, int(halfHeight-5))
            rightPoint = (pixelX, int(halfHeight+5))
            cv2.line(image, leftPoint, rightPoint, (0, 0, 0), 1)

    # MARK: > RENDER
    ## Render orthographic line-drawing projection for debugging
    def render(self, traceTrees=None):
        self.transformedScene = deepcopy(self.scene)
            
        for object in self.transformedScene.objects:
            object.setTransforms()
        
        ## Set up image
        image = np.zeros((self.options['height'],self.options['width'],3), np.uint8)
        image[:] = (255, 255, 255)
        
        halfWidth = self.options['width']/2
        halfHeight = self.options['height']/2
        
        ## Create BVHs for each object
        if self.options['bvh'] or self.options['aabb']:
            if self.options['debug']:
                print('building bvhs')
            for object in self.transformedScene.objects:
                object.bvh = object.buildBVH()
            if self.options['debug']:
                print('done')
        
        ## Draw axes and scale for positioning
        self.renderAxes(image, halfWidth, halfHeight)
        
        ## Render camera
        if self.options['camera']: self.renderCamera(image, halfWidth, halfHeight)
        
        for object in self.transformedScene.objects:
            ## Render BVH
            if object.type not in ['empty', 'camera']:
                if self.options['bvh']:
                    self.renderBVH(image, halfWidth, halfHeight, object.bvh, 20)
                elif self.options['aabb']:
                    self.renderBVH(image, halfWidth, halfHeight, object.bvh, 1)
            ## If object is sphere-based
            if (object.type == 'light' and self.options['lights']) or object.type == 'sphere':
                ## Get center position
                pixelVectorX, pixelVectorY = self.mapCoordinates(*object.position)
                pixelX = round((pixelVectorX / self.options['size'])+halfWidth)
                pixelY = round((pixelVectorY / self.options['size'])+halfHeight)
                radius = round((object.scale.x / 2) / self.options['size'])
                
                ## Draw sphere
                if object.type == 'light':
                    # cv2.circle(image, (pixelX, pixelY), 2, object.shader.debugColor.toTuple('bgr'), thickness=-1)
                    cv2.circle(image, (pixelX, pixelY), 2, object.shaders[0].debugColor.toTuple('bgr'), thickness=-1)
                    # cv2.circle(image, (pixelX, pixelY), radius, object.shader.debugColor.toTuple('bgr'), thickness=1)
                    cv2.circle(image, (pixelX, pixelY), radius, object.shaders[0].debugColor.toTuple('bgr'), thickness=1)
                else:
                    # cv2.circle(image, (pixelX, pixelY), radius, object.shader.debugColor.toTuple('bgr'), thickness=-1)
                    cv2.circle(image, (pixelX, pixelY), radius, object.shaders[0].debugColor.toTuple('bgr'), thickness=-1)
                
                continue
            ## If object is mesh-based
            elif object.type != 'mesh': continue
            if self.options['debug']:
                print(object.position)
            for face in object.mesh.faces:
                for vertexIndex in face:
                    ## Transform vertex
                    # vertex = object.mesh.vertices[vertexIndex].applyTransforms(object.position, object.scale, object.rotation)
                    vertex = object.mesh.vertices[vertexIndex]
                    
                    ## Check vertex is within bounds
                    if self.options['size']*halfWidth*-1 < vertex.x < self.options['size']*halfWidth:
                        if self.options['size']*halfHeight*-1 < vertex.z < self.options['size']*halfHeight:
                            ## Get vertex position
                            vertexX, vertexY = self.mapCoordinates(*vertex)
                            pixelX = round((vertexX / self.options['size'])+halfWidth)
                            pixelY = round((vertexY / self.options['size'])+halfHeight)
                            
                            ## Draw vertices
                            # cv2.circle(image, (pixelX, pixelY), 2, object.shader.debugColor.toTuple('bgr'), thickness=-1)
                            cv2.circle(image, (pixelX, pixelY), 2, object.shaders[object.shaderIndices[object.mesh.faces.index(face)]].debugColor.toTuple('bgr'), thickness=-1)

                            for otherVertexIndex in face:
                                ## Get other vertex position
                                # otherVertex = object.mesh.vertices[otherVertexIndex].applyTransforms(object.position, object.scale, object.rotation)
                                otherVertex = object.mesh.vertices[otherVertexIndex]

                                otherVertexX, otherVertexY = self.mapCoordinates(*otherVertex)
                                otherVertexPixelX = round((otherVertexX / self.options['size'])+halfWidth)
                                otherVertexPixelY = round((otherVertexY / self.options['size'])+halfHeight)
                                
                                ## Draw line between both vertices
                                # cv2.line(image, (pixelX, pixelY), (otherVertexPixelX, otherVertexPixelY), object.shader.debugColor.toTuple('bgr'), thickness=1)
                                cv2.line(image, (pixelX, pixelY), (otherVertexPixelX, otherVertexPixelY), object.shaders[object.shaderIndices[object.mesh.faces.index(face)]].debugColor.toTuple('bgr'), thickness=1)
        
        ## Render traces
        if traceTrees != None:
            for traceTree in traceTrees:
                self.renderTraceNode(image, traceTree, halfWidth, halfHeight, (0, 80, 200))
        
        ## Convert image data to Rendered and return
        # imageList = [row[::-1] for row in image.tolist()] ## FLIP COLORS?
        return Rendered(image, {}, self.options['width'], self.options['height'], options=self.options, failed=False, engine='debug')
    
    # MARK: > WRITE INFO
    ## Write trace info onto image
    def writeInfoToImg(self, img, trace, tracer):
        # print(trace)
        
        ## Define information
        objName = trace.collisionInfo['object'].name
        objType = trace.collisionInfo['object'].type
        directLight = tracer.getDirectLight(trace.collisionInfo)
        face = trace.collisionInfo['face']
        face_index = None
        
        ## Create text
        text = f'Name: {objName}\nType: {objType}\nDirect Light: {directLight}'
        if face != None and objType == 'mesh':
            print(face)
            face_index = trace.collisionInfo['object'].mesh.faces.index(face)
            text += f'\nFace: {face}\nFace Index: {face_index}'
        lines = text.split('\n')

        position = (10, 30)

        ## Create font
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        font_thickness = 1
        font_color = (0, 0, 0)
        
        ## Draw information
        for i, line in enumerate(lines):
            y_position = position[1] + i * 20
            cv2.putText(img, line, (position[0], y_position), font, font_scale, font_color, font_thickness)
        
        return img
    
    # MARK: > INSPECT
    ## Inspect a Rendered object to examine the traces within it
    def inspect(self, rendered, scene):
        tracerOptions = {}
        if rendered.engine == 'photon': tracerOptions = rendered.options
        
        tracer = Photon(scene=scene, options=tracerOptions)
        ## On click
        
        def click_event(event, x, y, _, __):
            if event == cv2.EVENT_LBUTTONDOWN:
                traces = []
                ## Find the correct 'pixel' according to how many actual pixels there are per 'pixel'
                if rendered.traces != {}:
                    for (px, py) in rendered.traces.keys():
                        if px <= x < px + rendered.options['step'] and py <= y < py + rendered.options['step']: # TODO # Sort this out
                            traces = rendered.traces[(px, py)]
                            break
                else:
                    pixelVec = tracer.pixelToVector(x, y)
                    ray = tracer.getRay(pixelVec)

                    traces = [tracer.trace(ray, bounces=rendered.options['bounces'], lights=rendered.options['lights'])]
                        
                originalDirection = self.options['direction']
                
                ## Render an orthographic projection for each view
                self.options['direction'] = 'top'
                topImg = self.render(traces).imageAs('cv2')
                cv2.imshow('debugTop', self.writeInfoToImg(topImg, traces[0], tracer))
                
                self.options['direction'] = 'side'
                sideImg = self.render(traces).imageAs('cv2')
                cv2.imshow('debugSide', self.writeInfoToImg(sideImg, traces[0], tracer))
                
                # self.options['direction'] = 'front'
                # frontImg = self.render(rendered.traces[(px, py)]).imageAs('np')
                # cv2.imshow('debugFront', self.writeInfoToImg(frontImg, rendered.traces[(px, py)][0]))
                
                ## Restore original direction
                self.options['direction'] = originalDirection
                
                print(traces)
        
        ## Display clickable render
        img = rendered.imageAs('cv2')
        cv2.imshow('img', img)
        cv2.setMouseCallback('img', click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# MARK: PHOTON
# Ray-tracing render engine
class Photon:
    def __init__(self, scene, options={}) -> None:
        self.scene = scene
        self.transformedScene = self.scene
        self.options = self.mergeDefaultOptions(options)
        self.rays = 0
    
    # MARK: > ADMIN
    ## Make sure all options exist even when unspecified
    def mergeDefaultOptions(self, options):
        defaultOptions = {'debug': False, 'showProgress': True, 'bounces': 1, 'wait': True, 'samples': 1, 'ambient': RGB(20, 20, 20), 'bvh': True, 'debugOptions': {}, 'aabb': True, 'bvh': True, 'lights': True, 'debug': False, 'emissionSampleDensity': 4, 'transformBefore': True, 'threads': 4, 'width': 1440, 'height': 900}
        for optionKey in defaultOptions.keys():
            if optionKey not in options.keys():
                options[optionKey] = defaultOptions[optionKey]
        
        return options

    # MARK: > RAY CALC
    ## Get a pixel's position on image plane
    def pixelToVector(self, pixelX, pixelY) -> Vec3:
        camera = self.transformedScene.camera
        
        # ## Correct flipped image
        # pixelX = self.options['width']-pixelX

        ## Get pixel position untransformed
        pixelLocalX = ((pixelX/self.options['width']) * (self.options['width']/self.options['height'])) - ((self.options['width']/self.options['height'])/2)
        pixelLocalY = 0.5 - (pixelY/self.options['height'])
        
        ## Transform position
        pixelAbsoluteVec = Vec3(pixelLocalX, pixelLocalY, camera.length/20).applyTransforms(camera.position, camera.scale, camera.rotation)
        
        return pixelAbsoluteVec
    
    ## Calculate initial ray
    def getRay(self, pixelVec) -> tuple[Vec3, Vec3]:
        rayStart = self.transformedScene.camera.position
        rayDirection = (pixelVec - rayStart).normalise()
        
        return Ray(rayStart, rayDirection, 1, None)
    
    # MARK: > FACEPLANE
    ## Convert face (vertices) to mathematical plane (point + normal)
    def getFacePlane(self, vertices):
        firstVec = vertices[1] - vertices[0]
        secondVec = vertices[2] - vertices[0]
        
        ## Cross vectors to get plane normal
        normalVec = firstVec.cross(secondVec)
        
        facePlane = MathematicalPlane(vertices[0], normalVec.normalise())
        
        return facePlane
    
    # MARK: > BVH FACES
    ## (Recursive) Use BVH structure to go retrieve the relevant faces to check against a ray
    def getFaces(self, ray, bvh):
        ## Base cases
        if not ray.entersAABB(bvh.box):
            return []
        if bvh.faces:
            return bvh.faces

        leftFaces = None
        rightFaces = None

        ## Recursion
        if bvh.left:
            leftFaces = self.getFaces(ray, bvh.left)
            # print('leftFaces:', leftFaces != [])
        if bvh.right:
            rightFaces = self.getFaces(ray, bvh.right)
            # print('rightFaces:', rightFaces != [])

        # return rightFaces if rightFaces is not [] else leftFaces
        return leftFaces + rightFaces

    # MARK: > COLLISION
    ## Find the closest collision of a ray with an object
    def getCollision(self, ray, lights, backCull=True):
        ## Default values to None
        closestCollision = None
        collisionObj = None
        collisionNormal = None
        collisionFace = None
        collisionDirection = None
        
        for object in self.transformedScene.objects:
            ## Check if ray enters object AABB
            if self.options['aabb'] and object.type in ['sphere', 'light', 'mesh']:
                bvh = object.bvh
                
                if not ray.entersAABB(bvh.box):
                    continue
            
            ## If object is sphere-based
            if object.type == 'sphere' or (lights and object.type == 'light'):
                ## Setup sphere object based on properties
                sphere = MathematicalSphere(object.position, object.scale.x/2)
                
                ## Get front and back collision distances
                ts = ray.getTSphere(sphere)
                if ts not in [None, [None, None]]:
                    t1, t2 = ts
                    
                    ## Collision with front face
                    if backCull or object.type == 'light':
                        t = t1
                    ## Collision with back face
                    else:
                        t = t2
                    
                    ## If collision is behind camera
                    if t < 0:
                        continue

                    pointOnSphere = ray.pointOnRay(t)
                    
                    normal = (pointOnSphere-sphere.position).normalise()
                    
                    ## If it is the first or closest collision
                    if closestCollision == None or (pointOnSphere - ray.start).magnitude() < (closestCollision - ray.start).magnitude():
                        closestCollision = pointOnSphere
                        collisionObj = object
                        collisionNormal = normal
                        if ray.direction.dot(normal) < 0:
                            collisionDirection = 0
                        else:
                            collisionDirection = 1
            ## If object is mesh-based
            elif object.type == 'mesh':
                # print(object.name)
                ## Get relevant faces
                if self.options['bvh']:
                    faces = self.getFaces(ray, object.bvh)
                else:
                    faces = object.mesh.faces

                for face in  faces:
                    if not self.options['transformBefore']:
                        ## Transform face vertices
                        faceVerticesLocal = [object.mesh.vertices[face[i]] for i in range(0, 3)]
                        
                        faceVertices = []
                        
                        for vertex in faceVerticesLocal:
                            vertexAbsolute = vertex.applyTransforms(object.position, object.scale, object.rotation)
                            faceVertices.append(vertexAbsolute)
                    else:
                        ## Get already-transformed faces
                        faceVertices = [object.mesh.vertices[face[i]] for i in range(0, 3)]
                    ## Get face plane
                    facePlane = self.getFacePlane(faceVertices)
                    
                    if ray.direction.perpendicular(facePlane.normal):
                        continue
                    else:
                        ## Get collision distance
                        t = ray.getT(facePlane)
                        
                        ## If collision is behind camera
                        if t < 0:
                            continue
                        
                        pointOnPlane = ray.pointOnRay(t)
                        
                        ## Check if collision occurs within triangle or without
                        
                        ## Vectors between vertices anticlockwise
                        triangle0To1 = faceVertices[1]-faceVertices[0]
                        triangle1To2 = faceVertices[2]-faceVertices[1]
                        triangle2To0 = faceVertices[0]-faceVertices[2]
                        
                        ## Vectors between vertices and collision point
                        triangle0ToPoint = pointOnPlane - faceVertices[0]
                        triangle1ToPoint = pointOnPlane - faceVertices[1]
                        triangle2ToPoint = pointOnPlane - faceVertices[2]
                        
                        ## Cross products
                        cross0 = triangle0To1.cross(triangle0ToPoint)
                        cross1 = triangle1To2.cross(triangle1ToPoint)
                        cross2 = triangle2To0.cross(triangle2ToPoint)

                        ## If all cross products face the same direction, collision occurs within the triangle
                        if (facePlane.normal.dot(cross0) < 0) == (facePlane.normal.dot(cross1) < 0) == (facePlane.normal.dot(cross2) < 0):
                            ## Ignore back faces if backCull is True
                            if ray.direction.dot(facePlane.normal) < 0 or not backCull:
                                if ray.direction.dot(facePlane.normal) < 0:
                                    collisionDirection = 0
                                else:
                                    collisionDirection = 1
                                
                                ## If it is the first or closest collision
                                if closestCollision == None or (pointOnPlane - ray.start).magnitude() < (closestCollision - ray.start).magnitude():
                                    closestCollision = pointOnPlane
                                    collisionObj = object
                                    collisionNormal = facePlane.normal
                                    collisionFace = face

        return {
            'coordinate': closestCollision,
            'object': collisionObj,
            'normal': collisionNormal,
            'direction': collisionDirection,
            'face': collisionFace,
        }
    
    # MARK: > BOUNCE
    ## Get the ray after bouncing off a surface
    def indirectBounce(self, ray, collisionInfo) -> Ray:
        ## Define information
        bouncePoint = collisionInfo['coordinate']
        obj = collisionInfo['object']
        normal = collisionInfo['normal'].normalise()

        ## Get a perfectly diffusive reflection; using multiple samples accounts for the randomness
        diffuseDirection = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
        diffuseDirection = diffuseDirection.normalise()
        if diffuseDirection.dot(normal) < 0:
            diffuseDirection = diffuseDirection * -1
        
        ## Get a perfectly specular reflection
        specularDirection = ray.direction.flipComponent(normal)

        # bounceDirection = (specularDirection * (1-obj.shader.roughness)) + (diffuseDirection * obj.shader.roughness)
        
        ## Get roughness of the face
        face = collisionInfo['face']
        roughness = 1
        if obj.type == 'mesh':
            roughness = obj.shaders[obj.shaderIndices[obj.mesh.faces.index(face)]].roughness
        elif obj.type in ['empty', 'sphere']:
            roughness = obj.shaders[0].roughness
        
        ## Combine diffusive and specular reflections to calculate ray
        bounceDirection = (specularDirection * (1-roughness)) + (diffuseDirection * roughness)
        bounceStart = bouncePoint + bounceDirection*0.1
        returnRay = Ray(bounceStart, bounceDirection, ray.ior, ray.object)

        return returnRay
    
    # MARK: > SHADOW
    ## Get the visibility of a ray to a light source to calculate shadows
    def visibility(self, ray) -> float:
        ## Get closest collision
        collisionInfo = self.getCollision(ray, True, False)
        collisionObj = collisionInfo['object']
        collisionFace = collisionInfo['face']
        
        if collisionObj != None and collisionObj.type == 'mesh' and self.options['debug']:
            print('rayStart:', ray.start)
            print('collisionPoint:', collisionInfo['coordinate'])
            print('name:', collisionObj.name)
            print('face:', collisionObj.mesh.faces.index(collisionFace))
            print('emission:', collisionObj.shaders[collisionObj.shaderIndices[collisionObj.mesh.faces.index(collisionFace)]].emissionStrength)
            print('reflectivity:', collisionObj.shaders[collisionObj.shaderIndices[collisionObj.mesh.faces.index(collisionFace)]].reflectivity)
        
        ## Check if the closest collision(?) is not a light or an emissive surface, i.e. the light is blocked by an object
        if (
            collisionObj != None
            and (
                collisionObj.type != 'light'
                and not (
                    collisionObj.type == 'mesh'
                    and collisionObj.shaders[collisionObj.shaderIndices[collisionObj.mesh.faces.index(collisionFace)]].emissionStrength > 0
                )
            )
        ):
            # visibility = 1-collisionObj.shader.reflectivity
            
            ## Get reflectivity
            reflectivity = 1
            if collisionObj.type == 'mesh':
                reflectivity = collisionObj.shaders[collisionObj.shaderIndices[collisionObj.mesh.faces.index(collisionFace)]].reflectivity
            elif collisionObj.type in ['empty', 'sphere']:
                reflectivity = collisionObj.shaders[0].reflectivity

            ## Calculate visbility based on how opaque an object is
            visibility = 1-reflectivity
            return visibility
        else:
            ## Completely exposed to light source
            return 1

    # MARK: > ATTENUATION
    ## Calculate the attenuation (falloff) of a light over distance
    def getAttenuation(self, distance, k=0.01) -> float:
        attenuation = 1 / (1 + k * distance ** 2)

        return max(0, min(1, attenuation))
    
    # MARK: > TRI AREA
    ## Calculate the area of a triangle using the cross product method
    def triangleArea(self, face):
        vertex1, vertex2, vertex3 = face
        
        ## Calculate edge vectors
        edge1 = vertex2 - vertex1
        edge2 = vertex3 - vertex1
        
        ## Calculate the cross product
        crossProduct = edge1.cross(edge2)
        
        ## Take half the magnitude as the area of the triangle
        area = 0.5 * crossProduct.magnitude()
        
        return area
    
    # MARK: > SAMPLE POINTS
    ## Calculate evenly distributed sample points within a triangle
    def getSamplePoints(self, face, vertices, density):
        ## Setup empty sample points list
        samplePoints = []
        
        ## Get vertex vectors
        v0, v1, v2 = [vertices[i] for i in face]
        
        ## Calculate the number of samples to take based on the density
        numSamples = math.ceil(density * self.triangleArea([v0, v1, v2]))
        
        if self.options['debug']:
            print('numSamples:', numSamples)

        for _ in range(numSamples):
            ## Take 2 random numbers between 0 and 1
            r1, r2 = random.random(), random.random()
            sqrtR1 = math.sqrt(r1)
            
            ## Calculate barycentric weights
            ## Using calculations involving sqrtR1 and r2 (instead of just r1, r2, r3) as our weights, we get a more uniform distribution throughout the triangle
            u = 1 - sqrtR1
            v = sqrtR1 * (1 - r2)
            w = sqrtR1 * r2

            ## Convert to 3D cartesian coordinates
            samplePoint = (v0 * u + v1 * v + v2 * w)

            samplePoints.append(samplePoint)
        
        return samplePoints
    
    # MARK: > DIRECT LIGHT
    ## Calculate the direct light intensity at a collision point
    def getDirectLight(self, collisionInfo) -> float | int:
        totalColor = RGB(0, 0, 0)
        totalIntensity = 0
        bouncePoint = collisionInfo['coordinate']
        lights = []
        for object in self.transformedScene.objects:
            ## If object emits light
            if object.type == 'light' or (object.type == 'mesh' and any(shader.emissionStrength > 0 for shader in object.shaders)):
                ## If object is emissive mesh
                if object.type == 'mesh':
                    for shader in object.shaders:
                        if shader.emissionStrength > 0:
                            for faceIndex in range(len(object.mesh.faces)):
                                # faceIndex = object.shaderIndices.index(object.shaders.index(shader))
                                if object.shaders[object.shaderIndices[faceIndex]] == shader:
                                    if self.options['debug']:
                                        print('face:', faceIndex)
                                    face = object.mesh.faces[faceIndex]
                                    strength = shader.emissionStrength / math.ceil((self.options['emissionSampleDensity'] * self.triangleArea([object.mesh.vertices[i] for i in face])))
                                    color = shader.color
                                    # pos = Mesh.centroid(object.mesh.vertices, face)
                                    for pos in self.getSamplePoints(face, object.mesh.vertices, self.options['emissionSampleDensity']):
                                        lights.append([pos, strength, color])
                ## If object is a light
                else:
                    pos = object.position
                    strength = object.strength
                    color = object.shaders[0].color
                    lights.append([pos, strength, color])
        
        for light in lights:
            pos = light[0]
            strength = light[1]
            color = light[2]
            if self.options['debug']:
                print('pos:', pos)
                print('strength:', strength)
                print('color:', color)
            
            ## Get the ray bouncing from the collision to the light
            bounceDirection = (pos-bouncePoint).normalise()
            bounceStart = bouncePoint + bounceDirection*0.2
            bounceRay = Ray(bounceStart, bounceDirection, 1, None)
            self.rays += 1
            
            ## Unit vector of the bounce ray and collision normal
            lightVector = (bounceRay.direction).normalise()
            normal = collisionInfo['normal'].normalise()
            # print(bounceRay, normal)

            ## Get intensity of light on a perfectly diffusive material
            diffuseIntensity = max((0, lightVector.dot(normal)))
            
            ## Get reflection vector and view vector
            reflectionVector = ((normal * lightVector.dot(normal) * 2) - lightVector)
            viewVector = (self.transformedScene.camera.position - collisionInfo['coordinate'])
            
            ## Get intensity of light on a perfectly specular material
            specularIntensity = max((0, (0 if reflectionVector.normalise().dot(viewVector.normalise()) < 0.96 else 1)))
            # if collisionInfo['object'].name == 'Mirror Ball':
            #     print('specularIntensity:', specularIntensity)
            
            ## Get roughness of material at collision point
            # roughness = collisionInfo['object'].shader.roughness
            roughness = 1
            if collisionInfo['object'].type == 'mesh':
                roughness = collisionInfo['object'].shaders[collisionInfo['object'].shaderIndices[collisionInfo['object'].mesh.faces.index(collisionInfo['face'])]].roughness
            elif collisionInfo['object'].type in ['empty', 'sphere']:
                roughness = collisionInfo['object'].shaders[0].roughness
            # roughness = collisionInfo['object'].shaders[collisionInfo['object'].mesh.faces.index(collisionInfo['face'])].roughness
            
            ## Get visibility to determine shadows
            visibility = self.visibility(bounceRay)
            
            ## Get light attenuation factor
            lightDistance = (pos-bouncePoint).magnitude()
            lightAttenuation = self.getAttenuation(lightDistance)
            
            ## Combine all factors to determine intensity of light at point, and add to total intensity of all lights
            intensity = ((diffuseIntensity*roughness) + (specularIntensity*(1-roughness))) * strength * visibility * lightAttenuation
            
            if self.options['debug']:
                print(
f'''\ndiffuse: {diffuseIntensity}
roughness: {roughness}
specular: {specularIntensity}
strength: {strength}
visbility: {visibility}
attenuation: {lightAttenuation}
intensity: {intensity}
color: {color}\n\n''')
            
            totalColor += color * 0.6
            totalIntensity += intensity
            
            # ## Cap totalIntensity to avoid intensity being too bright
            # if totalIntensity > 4:
            #     totalIntensity = 4

        return totalColor, totalIntensity
    
    # MARK: > TRANSMIT
    ## Transmit a ray using appropriate refraction techniques
    def transmit(self, ray, collisionInfo) -> Ray:
        # return Ray((collisionInfo['coordinate'] + (ray.direction * 0.2)), ray.direction)
        ## Get information for refraction
        normal = collisionInfo['normal']
        oldObject = ray.object
        oldIOR = ray.ior
        newObject = collisionInfo['object']
        newIOR = newObject.material.ior
        # newIOR = newObject.shader.ior
        
        ## Flip information if exiting an object instead of entering
        if newObject == oldObject:
            normal = normal * -1
            newIOR = 1
            newObject = None
        
        ## Snell's law calculations
        cosThetaI = normal.dot(ray.direction) * -1
        sinThetaI2 = max(0.0, 1.0 - (cosThetaI ** 2))
        ratio = oldIOR / newIOR
        sinThetaT2 = (ratio**2) * sinThetaI2
        
        ## Return if total internal reflection occurs
        if sinThetaT2 > 1.0:
            return None

        cosThetaT = math.sqrt(1.0 - sinThetaT2)

        ## Get a perfectly specular refraction
        specularDirection = (ray.direction * ratio) + (normal * ((ratio * cosThetaI) - cosThetaT))

        ## Get a perfectly diffusive refraction; using multiple samples accounts for the randomness
        diffuseDirection = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
        diffuseDirection = diffuseDirection.normalise()
        if diffuseDirection.dot(normal) > 0:
            diffuseDirection = diffuseDirection * -1
        
        ## Get roughness of the face
        face = collisionInfo['face']
        roughness = 1
        if collisionInfo['object'].type == 'mesh':
            roughness = collisionInfo['object'].shaders[collisionInfo['object'].shaderIndices[collisionInfo['object'].mesh.faces.index(face)]].roughness
        elif collisionInfo['object'].type in ['empty', 'sphere']:
            roughness = collisionInfo['object'].shaders[0].roughness
        
        ## Combine diffusive and specular refractions to calculate ray
        bounceDirection = (specularDirection * (1-roughness)) + (diffuseDirection * roughness)
        bounceStart = collisionInfo['coordinate'] + bounceDirection*0.1

        refractedRay = Ray(bounceStart, bounceDirection, ray.ior, ray.object)
        
        # refractedRay = Ray((collisionInfo['coordinate'] + (refractedDirection * 0.1)), refractedDirection, newIOR, newObject)
        
        return refractedRay
    
    # MARK: > TRACE
    ## (Recursive) Trace a ray's path until it reaches a breakpoint
    def trace(self, ray, bounces, lights=False) -> TraceTreeNode | None:
        ## Get collision and node info
        collisionInfo = self.getCollision(ray, lights, ray.ior == 1)
        node = TraceTreeNode(ray, collisionInfo)
        # print(collisionInfo)
        
        ## Base cases
        if collisionInfo['coordinate'] == None:
            return None
        elif collisionInfo['object'].type == 'light' or (collisionInfo['object'].type == 'mesh' and collisionInfo['object'].shaders[collisionInfo['object'].shaderIndices[collisionInfo['object'].mesh.faces.index(collisionInfo['face'])]].emissionStrength > 0):
            return node

        elif bounces > 1:
            ## Get reflectivity of material at collision point
            reflectivity = 1
            if collisionInfo['object'].type == 'mesh':
                reflectivity = collisionInfo['object'].shaders[collisionInfo['object'].shaderIndices[collisionInfo['object'].mesh.faces.index(collisionInfo['face'])]].reflectivity
            elif collisionInfo['object'].type in ['empty', 'sphere']:
                reflectivity = collisionInfo['object'].shaders[0].reflectivity
            
            ## Not perfectly opaque
            if reflectivity != 1:
                ## Calculate transmitted ray
                transmitRay = self.transmit(ray, collisionInfo)
                if transmitRay != None:
                    transmitNode = self.trace(transmitRay, bounces-1, lights=True)
                else:
                    transmitNode = None
            else:
                transmitNode = None
            ## Not perfectly transparent
            if reflectivity != 0:
                ## Calculate reflected ray
                bounceRay = self.indirectBounce(ray, collisionInfo)
                if bounceRay != None:
                    bounceNode = self.trace(bounceRay, bounces-1, lights=True)
                else:
                    bounceNode = None
            else:
                bounceNode = None
            
            ## Chain nodes to create trace tree
            node.left = bounceNode
            node.right = transmitNode

        return node

    # MARK: > COLOR
    ## (Recursive) Compute the color of a pixel given its trace
    def computeColor(self, trace) -> RGB:
        ## Base case
        if trace == None:
            return self.options['ambient']
        
        obj = trace.collisionInfo['object']
        
        ## Object is light
        if obj.type == 'light':
            ## Get light color and strength
            # color = obj.shader.color
            color = obj.shaders[0].color
            color *= obj.strength
            
            ## Multiply by attenuation
            distance = (trace.collisionInfo['coordinate'] - self.transformedScene.camera.position).magnitude()
            directAttenuation = self.getAttenuation(distance)
            color *= directAttenuation
            
            return color.clamp()

        ## Object is emissive mesh
        if obj.type == 'mesh' and obj.shaders[obj.shaderIndices[obj.mesh.faces.index(trace.collisionInfo['face'])]].emissionStrength > 0:
            ## Get face color and strength
            # color = obj.shader.color
            color = obj.shaders[obj.shaderIndices[obj.mesh.faces.index(trace.collisionInfo['face'])]].color
            color *= obj.shaders[obj.shaderIndices[obj.mesh.faces.index(trace.collisionInfo['face'])]].emissionStrength
            
            ## Multiply by attenuation
            distance = (trace.collisionInfo['coordinate'] - self.transformedScene.camera.position).magnitude()
            directAttenuation = self.getAttenuation(distance)
            color *= directAttenuation
            
            return color.clamp()
        
        ## Get shader from object
        # shader = obj.shader
        if obj.type == 'mesh':
            shader = obj.shaders[obj.shaderIndices[obj.mesh.faces.index(trace.collisionInfo['face'])]]
        elif obj.type in ['empty', 'camera', 'sphere']:
            shader = obj.shaders[0]
        else:
            raise TypeError(f'Could not recognise object type {trace.collisionInfo["object"].type}')
        
        color = shader.color

        ## Recursion to get colors from next rays
        reflectedColor = self.computeColor(trace.left) if trace.left is not None else self.options['ambient'] * (2/255)
        transmittedColor = self.computeColor(trace.right) if trace.right is not None else self.options['ambient'] * (2/255)
        
        ## Factors for combination
        reflectivity = shader.reflectivity
        transmissionFactor = 1 - reflectivity
        
        ## Get attenuations
        reflectedDistance = (trace.left.collisionInfo['coordinate']-trace.ray.start).magnitude() if trace.left is not None else 0
        reflectedAttenuation = self.getAttenuation(reflectedDistance)        
        transmittedDistance = (trace.right.collisionInfo['coordinate']-trace.ray.start).magnitude() if trace.right is not None else 0
        transmittedAttenuation = self.getAttenuation(transmittedDistance)
        
        # print(reflectedAttenuation, transmittedAttenuation)

        ## Combine factors to calculate indirect color
        indirectColor = (reflectedColor * reflectivity * reflectedAttenuation) + (transmittedColor * transmissionFactor * transmittedAttenuation * 1.4)
        
        ## Get direct color
        directInitialColor, directIntensity = self.getDirectLight(trace.collisionInfo)
        directColor = ((color * directIntensity) + (directInitialColor * directIntensity * (2/255))) * reflectivity
        # print('directIntensity:', directIntensity)

        ## Add indirect, direct and base ambient color
        color = indirectColor + directColor + (color * self.options['ambient'] * (1/255))

        if self.options['debug']:
            print(f'reflected: {reflectedColor}\ntransmitted: {transmittedColor}\nindirect: {indirectColor}\ndirect: {directColor}\ncolor: {color}\n')

        return color.clamp()
    
    # MARK: > MERGE IMGS
    ## Merge images using mask
    # def mergeImgs(self, imgs, resWidth, resHeight):
    #     ## Setup the empty final image
    #     totalImg = np.zeros((resHeight, resWidth, 4), np.uint8)

    #     ## Iterate images
    #     for img in imgs:
    #         mask = img[..., 3] == 255

    #         totalImg = np.where(mask[..., None], img, totalImg)
        
    #     return totalImg
    
    def mergeImgs(self, stripes):
        # img = stripes[0]
        # stripes = stripes[1:]
        
        # for stripe in stripes:
        #     img = np.concatenate((img, stripe))
        
        img = np.vstack(stripes)
        # img = np.concatenate(stripes, axis=0)
        
        return img
    
    # MARK: > RENDER
    ## Create a Rendered object based on the scene properties and data
    def render(self, progressCallback=lambda _, __, ___: None, cancelCallback=lambda: False) -> Rendered:
        print('beginning setup')
        
        ## Copy the scene so object transforms can be applied without repercussions
        self.transformedScene = deepcopy(self.scene)
        
        ## Check bound camera exists
        if not self.transformedScene.camera:
            raise NoBoundCameraError()
        
        if self.options['transformBefore']:
            ## Set transforms (optimisation - set them at the start instead of applying them for every calculation)
            for object in self.transformedScene.objects:
                if object.type == 'mesh':
                    object.setTransforms()
            
            print('applied transforms')

        if self.options['bvh']:
            ## Build BVHs (optimisation)
            for object in self.transformedScene.objects:
                object.bvh = object.buildBVH()

            print('built BVHs')

        ## TODO # Use a start and end row index depending on threads
        ## TODO # Merge with blocks as pixelOrder is unnecessary without step
        if self.options['height'] % self.options['threads'] != 0:
            raise ValueError('Number of threads must be a factor of image height')
        
        blockSize = self.options['height'] // self.options['threads']
        # ## Generate the pixels to render based on render modes and resolution
        # pixelOrderX = list(range(0, self.options['width'], (self.options['step'])))
        # pixelOrderY = list(range(0, self.options['height'], (self.options['step'])))
        # pixelOrder = []

        # for i in pixelOrderX:
        #     for j in pixelOrderY:
        #         pixelOrder.append((i, j))

        ## Store information for Rendered object later
        totalTraces = {}
        self.rays = 0
        start = time.time()

        ## TODO # This can go with new pixel order system
        # blockSize = math.ceil(len(pixelOrder)/self.options['threads'])
        # blocks = [pixelOrder[i:i+blockSize] for i in range(0, len(pixelOrder), blockSize)]

        renderQueue = Queue()
        threads = []
        
        self.threadsProgress = ThreadsProgress(callback=progressCallback, total=blockSize*self.options['width'])

        # for threadId, block in enumerate(blocks):
        #     thread = threading.Thread(target=self.renderPixels, args=(block, cancelCallback, threadId, renderQueue))
        #     threads.append(thread)
        #     thread.start()
        for threadId in range(self.options['threads']):
            thread = threading.Thread(target=self.renderPixels, args=(blockSize, cancelCallback, threadId, renderQueue))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        print('finished rendering')
        
        renderResults = []
        while not renderQueue.empty():
            renderResult = renderQueue.get()
            renderResults.append(renderResult)
        
        renderResults.sort(key=lambda res: res[0])
        
        stripes = []
        failed = False
        totalTraces = {}
        for renderResult in renderResults:
            _, stripe, traces, threadFailed = renderResult
            stripes.append(stripe)
            totalTraces.update(traces)
            if threadFailed or failed:
                failed = True
            
        end = time.time()
        
        # image = self.mergeImgs(images, self.options['width'], self.options['height'])
        image = self.mergeImgs(stripes)

        print('finished merging')

        ## Set final Rendered properties and then return
        totalTime = end-start
        rays = self.rays
        self.rays = 0
        
        print(totalTime)
        
        return Rendered(image, totalTraces, self.options['width'], self.options['height'], totalTime, rays, self.options, failed=failed, engine='photon')
    
    # def renderPixels(self, pixelOrder, cancelCallback, id=None, queue=None):
    def renderPixels(self, blockSize, cancelCallback, id=None, queue=None):
        self.threadsProgress[id] = 0
        # if self.options['debug']:
        # print(f'Thread {id} started')
        ## Setup image
        # image = np.zeros((self.options['height'], self.options['width'], 3), np.uint8)
        stripe = np.zeros((blockSize, self.options['width'], 3), np.uint8)
        
        totalTraces = {}
        
        failed = False
        
        startTime = time.time()

        startRow = id*blockSize

        for pixelY in range(blockSize):
            for pixelX in range(self.options['width']):
                try:
                    ## Cast first ray
                    pixelVec = self.pixelToVector(pixelX, startRow+pixelY)
                    initialRay = self.getRay(pixelVec)
                    
                    ## Trace the initial ray several times for more accurate sampling
                    traces = []
                    samples = []
                    for _ in range(self.options['samples'] if self.options['bounces'] > 1 else 1):
                        rayTrace = self.trace(initialRay, bounces=self.options['bounces'], lights=self.options['lights'])
                        self.rays += 1
                        traces.append(rayTrace)
                        # if rayTrace != None: print(rayTrace)
                        samples.append(self.computeColor(rayTrace))

                    totalTraces[(pixelX, startRow+pixelY)] = traces
                    
                    ## Take a mean average of the traces colors
                    # color = RGB.mean(samples).toTuple('bgr')
                    color = RGB.mean(samples).toTuple('bgr')
                    
                    # print(samples)
                    # print(color)
                    
                    stripe[pixelY][pixelX] = color
                    
                    rowsCompleted = pixelY-startRow
                    progress = pixelX + (rowsCompleted*self.options['width'])
                    self.threadsProgress[id] = progress

                    if cancelCallback():
                        if self.options['debug']:
                            print('Safely cancelling render')
                        failed = True
                        break
                ## Catch keyboard interrupt and finish render, marking failed as True for Rendered object
                except KeyboardInterrupt:
                    if self.options['debug']:
                        print('Safely exiting render')
                    failed = True
                    break

        timeToComplete = time.time()-startTime
        # if self.options['debug']:
        # print(f'Thread {id} done in {timeToComplete:.2f}')
        queue.put([id, stripe, totalTraces, failed])


# MARK: THREADS PROGRESS
## Structure to allow tracking of threads progress
class ThreadsProgress: ## TODO # return just one total size, not all
    def __init__(self, callback, total) -> None:
        ## Lock prevents race conditions
        ## Using `with self.lock` will lock the SharedState while the with block is being executed
        self.lock = threading.Lock()
        self.total = total
        self.callback = callback
        self.__threadsProgress = {}
    
    # MARK: > GET
    def __getitem__(self, key):
        with self.lock:
            ## Return data
            return self.__threadsProgress[key]

    # MARK: > SET
    def __setitem__(self, key, value):
        with self.lock:
            ## Update data and run callback
            self.__threadsProgress[key] = value
            self.callback(self.__threadsProgress, key, self.total)


# MARK: TEXEL
## Simplified rasterisation engine for real-time previews
class Texel:
    def __init__(self, scene, options={}) -> None:
        self.scene = scene
        self.transformedScene = self.scene
        self.options = self.mergeDefaultOptions(options)

    # MARK: > ADMIN
    ## Make sure all options exist even when unspecified
    def mergeDefaultOptions(self, options):
        defaultOptions = {'attenuation': True, 'backCull': True, 'vertices': False, 'edges': False, 'selectedObject': -1, 'normals': False, 'ambient': RGB(0, 0, 0), 'width': 1440, 'height': 900}
        for optionKey in defaultOptions.keys():
            if optionKey not in options.keys():
                options[optionKey] = defaultOptions[optionKey]
        
        return options

    # MARK: > PROJECT
    ## Get vertex position on screen
    def projectVertex(self, vertex):
        camera = self.transformedScene.camera
        
        ## Apply camera transforms so that the camera is effectively at (0, 0, 0), rotated by (0, 0, 0)
        transformedVertex = vertex.applyTransforms(camera.position * -1, Scale(1, 1, 1), camera.rotation * -1)
        
        ## Check vertex is not behind camera
        if transformedVertex.z < 0.00001:
            return None
        
        ## Project the vertex onto the screen space
        scaleFac = (camera.length/20) / transformedVertex.z
        vecX = transformedVertex.x * scaleFac
        vecY = transformedVertex.y * scaleFac
        
        ## Convert vertex vector into pixel coordinates using screen width
        halfWidth = (self.options['width']/self.options['height']) / 2
        pixelX = ((halfWidth + vecX)/(self.options['width']/self.options['height'])) * self.options['width']
        pixelY = (0.5 - vecY) * self.options['height']
        
        ## Only return if pixel is actually on the desired image
        if pixelX > self.options['width'] or pixelX < 0 or pixelY > self.options['height'] or pixelY < 0:
            return None
        
        return (int(pixelX), int(pixelY))

    # MARK: RENDER
    ## Render a simple rasterised image of a scene with projection
    def render(self):
        print('beginning setup')
        
        ## Copy the scene so object transforms can be applied without repercussions
        self.transformedScene = deepcopy(self.scene)
        
        ## Check bound camera exists
        if not self.transformedScene.camera:
            raise NoBoundCameraError()
        
        ## Set transforms (optimisation - set them at the start instead of applying them for every calculation)
        for object in self.transformedScene.objects:
            if object.type == 'mesh':
                object.setTransforms()
        
        ## Setup a photon instance to assist with more accurate shading
        if self.options['attenuation'] or self.options['backcull']:
            photon = Photon(scene=self.transformedScene)

        ## Setup image to work on
        # image = np.zeros((self.options['height'], self.options['width'], 3), np.uint8)
        image = np.full((self.options['height'], self.options['width'], 3), self.options['ambient'].toTuple('bgr'), dtype=np.uint8)
        
        ## Start timing render
        start = time.time()

        for obj in self.transformedScene.objects:
            if obj == self.transformedScene.camera and self.options['selectedObject'] == obj.id and self.options['edges']:
                cv2.rectangle(image, (0, 0), (self.options['width']-1, self.options['height']-1), (0, 100, 255), 3)
            if obj.type == 'sphere':
                ## Project 3D vector to screenspace point
                projectedPoint = self.projectVertex(obj.position)
                
                if projectedPoint:
                    ## Calculate a screenspace radius by projecting a circumference point
                    projectedCircumferencePoint = self.projectVertex(obj.position + Vec3(obj.scale.x/2, 0, 0))
                    
                    if projectedCircumferencePoint:
                        dx = abs(projectedCircumferencePoint[0] - projectedPoint[0])
                        dy = abs(projectedCircumferencePoint[1] - projectedPoint[1])
                        screenspaceRadius = round(math.sqrt(dx**2 + dy**2))
                        
                        color = obj.shaders[0].color
                        shadedColor = color
                        if self.options['attenuation']:
                            ## Get attenuation
                            dist = (obj.position - self.transformedScene.camera.position).magnitude()
                            attenuation = photon.getAttenuation(dist)
                            
                            ## Get shaded color
                            shadedColor = color * attenuation
                        
                        cv2.circle(image, projectedPoint, screenspaceRadius, shadedColor.toTuple('bgr'), -1)
            elif obj.type == 'mesh':
                for face in obj.mesh.faces:
                    vertices = [
                        obj.mesh.vertices[index]
                        for index in face
                    ]
                    projected = [self.projectVertex(vertex) for vertex in vertices]
                    ## Make sure all vertices are on the screen and in front of the camera
                    if not any([vertex == None for vertex in projected]):
                        center = Mesh.centroid(vertices, [0, 1, 2])
                        facePlane = photon.getFacePlane(vertices)
                        if facePlane.normal.dot((self.transformedScene.camera.position - center)) < 0:
                            normalDirection = 'away'
                        else:
                            normalDirection = 'toward'

                        ## Cull any faces facing away
                        if self.options['backCull'] and normalDirection == 'away':
                            continue
                        
                        ## Display object color
                        color = obj.shaders[obj.shaderIndices[obj.mesh.faces.index(face)]].color

                        if self.options['normals']:
                            ## Display red/blue depending on normal direction
                            if normalDirection == 'away':
                                color = RGB(230, 0, 0)
                            elif normalDirection == 'toward':
                                color = RGB(0, 0, 230)

                        shadedColor = color

                        if self.options['attenuation']:
                            ## Get attenuation
                            center = Mesh.centroid(vertices, [0, 1, 2])
                            dist = (center - self.transformedScene.camera.position).magnitude()
                            attenuation = photon.getAttenuation(dist)
                            
                            ## Get shaded color
                            shadedColor = color * attenuation
                        ## Draw face between three vertices
                        cv2.fillPoly(image, np.int32([projected]), shadedColor.toTuple('bgr'))
                        
                        if self.options['edges']:
                            ## Draw edges
                            lineColor = (0, 0, 0)
                            ## Make selected object have orange edges
                            if obj.id == self.options['selectedObject']:
                                lineColor = (0, 100, 255)
                            cv2.polylines(image, np.int32([projected]), isClosed=True, color=lineColor, thickness=1)
                        
                        if self.options['vertices']:
                            ## Draw vertices
                            vertexColor = (0, 0, 0)
                            ## Make selected object have orange vertices
                            if obj.id == self.options['selectedObject']:
                                vertexColor = (0, 100, 255)
                            for vertex in projected:
                                cv2.circle(image, vertex, 1, color=vertexColor, thickness=-1)
            
        ## Calculate time to render
        end = time.time()
        totalTime = end-start
        
        return Rendered(image, {}, self.options['width'], self.options['height'], totalTime, [], self.options, failed=False, engine='texel')


class Post:
    @staticmethod
    def compressRenderedDict(rendered, tolerance=10):
        imgData = rendered['imgData']
        compressedData = []
        
        for row in imgData:
            compressedRow = []
            lastPixel = None
            for pixel in row:
                if lastPixel == None:
                    compressedRow.append([pixel, 1])
                    lastPixel = pixel
                    continue

                dr = pixel[0] - lastPixel[0]
                dg = pixel[1] - lastPixel[1]
                db = pixel[2] - lastPixel[2]
                
                d = math.sqrt(dr**2 + dg**2 + db**2)
                
                if d <= tolerance:
                    compressedRow[-1][1] += 1
                else:
                    compressedRow.append([pixel, 1])
                    lastPixel = pixel
            compressedData.append(compressedRow)
        
        newRendered = copy.deepcopy(rendered)
        newRendered['imgData'] = compressedData

        return newRendered
    
    @staticmethod
    def decompressRenderedDict(rendered):
        imgData = rendered['imgData']
        decompressedData = []
        
        for row in imgData:
            decompressedRow = []
            for pixel in row:
                for _ in range(pixel[1]):
                    decompressedRow.append(pixel[0])
            decompressedData.append(decompressedRow)
        
        newRendered = copy.deepcopy(rendered)
        newRendered['imgData'] = decompressedData

        return newRendered
    
    @staticmethod
    def upscale(imageData, factor):
        height, width, depth = imageData.shape
        
        newImageData = np.zeros((height*factor, width*factor, depth), np.uint8)
        
        for row in range(height):
            for col in range(width):
                pixel = imageData[row][col]
                for i in range(factor):
                    for j in range(factor):
                        newImageData[row * factor + i][col * factor + j] = pixel
        
        return newImageData
    
    @staticmethod
    def denoise(imageData, kernelSize):
        padding = kernelSize // 2

        height = len(imageData)
        width = len(imageData[0])

        denoisedImage = [[(0, 0, 0)] * width for _ in range(height)]

        def median(lst):
            sortedLst = sorted(lst)
            return sortedLst[len(sortedLst) // 2]

        for i in range(padding, height - padding):
            for j in range(padding, width - padding):
                neighbourhoodB = []
                neighbourhoodG = []
                neighbourhoodR = []

                for ki in range(-padding, padding + 1):
                    for kj in range(-padding, padding + 1):
                        b, g, r = imageData[i + ki][j + kj]
                        neighbourhoodB.append(b)
                        neighbourhoodG.append(g)
                        neighbourhoodR.append(r)

                medianB = median(neighbourhoodB)
                medianG = median(neighbourhoodG)
                medianR = median(neighbourhoodR)

                denoisedImage[i][j] = (medianB, medianG, medianR)

        return denoisedImage