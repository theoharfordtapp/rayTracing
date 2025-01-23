# AQA A-Level Computer Science NEA 2024
#
# Graphics Engine
# Theo HT
#
# (Rendering Engines)
#
# NOTE # Comments with `MARK` in them are purely IDE-related. They have no relevance to the code.

from T3DE import Vec3, Ray, MathematicalPlane, MathematicalSphere, RGB, TraceTreeNode, Scale, Mesh, Euler
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

## Custom error for when no camera is bound to the scene
class NoBoundCameraError(Exception):
    def __init__(self, message="Scene contains no bound camera. Bind a camera in order to render.\n\n\t\tScene().camera = Camera()\n\n\tOR\tcamera = Camera(Scene())\n\n"):
        self.message = message
        super().__init__(self.message)


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
            case 'tupleList':
                return self.imgData.astype(int).toList()
            case 'listList':
                return [[
                    [int(pix[0]), int(pix[1]), int(pix[2])]
                    for pix in row]
                    for row in self.imageAs('tupleList')
                ]
            case 'cv2':
                return np.flip(self.imgData, 2) ## Convert RGB to BGR
            case 'np':
                return self.imgData
    
    ## Convert rendered to dictionary
    def toDict(self):
        options = self.options
        if 'ambient' in options.keys():
            ## Convert RGB to list
            options['ambient'] = [*options['ambient']]
        return {
            'imgData': self.imageAs('listlist'),
            'traces': { ## Convert pixel tuple into string for json usage
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
    
    ## Convert dictionary to rendered
    @staticmethod
    def fromDict(data):
        options = data.get('options', {})
        
        if 'ambient' in options.keys():
            ## Convert list to RGB
            options['ambient'] = RGB(*options['ambient'])
        return Rendered(
            imgData=[[ ## Rendered expects BGR tuple list
                (int(pix[2]), int(pix[1]), int(pix[0]))
                for pix in row]
                for row in data.get('imgData', [])
            ],
            traces={ ## Convert pixel strings to tuples
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
    
    ## Save rendered to JSON file
    def saveJSON(self, filepath):
        data = self.toDict()
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    ## Load rendered from JSON file
    @staticmethod
    def fromJSON(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return Rendered.fromDict(data)


## Provide an orthographic line-drawing of the scene for debug purposes
class Debug:
    def __init__(self, scene, options={}) -> None:
        self.scene = scene
        self.transformedScene = self.scene
        self.options = self.mergeDefaultOptions(options)

    ## Make sure all options exist even when unspecified
    def mergeDefaultOptions(self, options):
        defaultOptions = {'size':1, 'width':1440, 'height':900, 'direction': 'top', 'lights': True, 'camera': True, 'aabb': True, 'bvh': False, 'debug': True}
        for optionKey in defaultOptions.keys():
            if optionKey not in options.keys():
                options[optionKey] = defaultOptions[optionKey]
        
        options['size'] /= 100
        
        return options

    ## Size needs to be much smaller number than is practical
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

    ## Display a ray trace
    def renderTraceNode(self, image, traceNode, halfWidth, halfHeight, color):
        if traceNode is None:
            return

        ray = traceNode.ray

        ## Get the ray start position
        rayVectorX, rayVectorY = self.mapCoordinates(*ray.start)
        rayStartPixelX = round((rayVectorX / self.options['size']) + halfWidth)
        rayStartPixelY = round((rayVectorY / self.options['size']) + halfHeight)
        
        if None not in traceNode.collisionInfo:
            collisionPoint = traceNode.collisionInfo['point']

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
            cv2.line(image, (rayEndPixelX, rayEndPixelY), (normalEndPixelX, normalEndPixelY), (255, 133, 255), thickness=1)
        else:
            ## Get an arbitrary 'infinite' (finite in reality) ray end position
            rayEndVectorX, rayEndVectorY = self.mapCoordinates(*(ray.start + (ray.direction*1000)))
            rayEndPixelX = round((rayEndVectorX / self.options['size']) + halfWidth)
            rayEndPixelY = round((rayEndVectorY / self.options['size']) + halfHeight)
            
            ## Draw ray
            cv2.line(image, (rayStartPixelX, rayStartPixelY), (rayEndPixelX, rayEndPixelY), color, thickness=1)

        ## Recursively render next bounces from ray
        if traceNode.left is not None:
            self.renderTraceNode(image, traceNode.left, halfWidth, halfHeight, (0, 80, 200))
        if traceNode.right is not None:
            self.renderTraceNode(image, traceNode.right, halfWidth, halfHeight, (200, 80, 0))

    ## Draw the camera
    def renderCamera(self, image, halfWidth, halfHeight):
        camera = self.transformedScene.camera
        
        ## Get camera position
        cameraVectorX, cameraVectorY = self.mapCoordinates(*camera.position)
        cameraPixelX = round((cameraVectorX / self.options['size']) + halfWidth)
        cameraPixelY = round((cameraVectorY / self.options['size']) + halfHeight)
        
        ## Draw camera focal point
        cv2.circle(image, (cameraPixelX, cameraPixelY), 2, camera.shaders[0].debugColor.toTuple('bgr'), thickness=-1)

        ## Get image plane vertices
        vertices = [
            (Vec3(0, 0, camera.length/20) + Vec3(-(self.options['width']/self.options['height']) / 2, 0.5, 0)).applyTransforms(camera.position, camera.scale, camera.rotation),
            (Vec3(0, 0, camera.length/20) + Vec3((self.options['width']/self.options['height']) / 2, 0.5, 0)).applyTransforms(camera.position, camera.scale, camera.rotation),
            (Vec3(0, 0, camera.length/20) + Vec3(-(self.options['width']/self.options['height']) / 2, -0.5, 0)).applyTransforms(camera.position, camera.scale, camera.rotation),
            (Vec3(0, 0, camera.length/20) + Vec3((self.options['width']/self.options['height']) / 2, -0.5, 0)).applyTransforms(camera.position, camera.scale, camera.rotation),
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
                    camera.shaders[0].debugColor.toTuple('bgr'),
                    thickness=1
                )

            ## Draw line from vertex to focal point
            cv2.line(image, (pixelX, pixelY), (cameraPixelX, cameraPixelY), camera.shaders[0].debugColor.toTuple('bgr'), thickness=1)

            directionX = pixelX - cameraPixelX
            directionY = pixelY - cameraPixelY

            extendedPixelX = pixelX + int(directionX / (directionX**2 + directionY**2)**0.5 * 400)
            extendedPixelY = pixelY + int(directionY / (directionX**2 + directionY**2)**0.5 * 400)

            ## Draw line outwards to show field of view
            cv2.line(image, (pixelX, pixelY), (extendedPixelX, extendedPixelY), (128, 128, 128), thickness=1)

    ## Draw BVH
    def renderBVH(self, image, halfWidth, halfHeight, bvh, maxDepth, depth=0):
        if bvh is None or bvh.box == None or depth == maxDepth:
            return

        boxMin = bvh.box[0]
        boxMax = bvh.box[1]

        ## Get pixels for each corner
        pixelMinVectorX, pixelMinVectorY = self.mapCoordinates(*boxMin)
        pixelMinX = round((pixelMinVectorX / self.options['size']) + halfWidth)
        pixelMinY = round((pixelMinVectorY / self.options['size']) + halfHeight)

        pixelMaxVectorX, pixelMaxVectorY = self.mapCoordinates(*boxMax)
        pixelMaxX = round((pixelMaxVectorX / self.options['size']) + halfWidth)
        pixelMaxY = round((pixelMaxVectorY / self.options['size']) + halfHeight)

        ## Draw rectangle
        cv2.rectangle(image, (pixelMinX, pixelMinY), (pixelMaxX, pixelMaxY), (200, 100, 200), 1)

        ## Recursively draw the next boxes
        self.renderBVH(image, halfWidth, halfHeight, bvh.left, 20)
        self.renderBVH(image, halfWidth, halfHeight, bvh.right, 20)

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
                            shader = object.getShader(face)
                            cv2.circle(image, (pixelX, pixelY), 2, shader.debugColor.toTuple('bgr'), thickness=-1)

                            for otherVertexIndex in face:
                                ## Get other vertex position
                                # otherVertex = object.mesh.vertices[otherVertexIndex].applyTransforms(object.position, object.scale, object.rotation)
                                otherVertex = object.mesh.vertices[otherVertexIndex]

                                otherVertexX, otherVertexY = self.mapCoordinates(*otherVertex)
                                otherVertexPixelX = round((otherVertexX / self.options['size'])+halfWidth)
                                otherVertexPixelY = round((otherVertexY / self.options['size'])+halfHeight)
                                
                                ## Draw line between both vertices
                                # cv2.line(image, (pixelX, pixelY), (otherVertexPixelX, otherVertexPixelY), object.shader.debugColor.toTuple('bgr'), thickness=1)
                                cv2.line(image, (pixelX, pixelY), (otherVertexPixelX, otherVertexPixelY), shader.debugColor.toTuple('bgr'), thickness=1)
        
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
                    if (x, y) in rendered.traces.keys():
                        traces = rendered.traces[(x, y)]
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
    
    @property
    def scene(self):
        return self.__scene
    
    @scene.setter
    def scene(self, scene):
        self.__scene = scene
        self.transformedScene = scene
    
    # MARK: > ADMIN
    ## Make sure all options exist even when unspecified
    def mergeDefaultOptions(self, options):
        defaultOptions = {'debug': False, 'bounces': 1, 'wait': True, 'samples': 1, 'ambient': RGB(20, 20, 20), 'bvh': True, 'debugOptions': {}, 'aabb': True, 'bvh': True, 'lights': True, 'debug': False, 'transformBefore': True, 'threads': 4, 'width': 1440, 'height': 900}
        for optionKey in defaultOptions.keys():
            if optionKey not in options.keys():
                options[optionKey] = defaultOptions[optionKey]
        
        return options

    ## Get a pixel's position on image plane
    def pixelToVector(self, pixelX, pixelY) -> Vec3:
        scene = self.transformedScene if self.options['transformBefore'] else self.scene
        camera = scene.camera

        width, height = self.options['width'], self.options['height']
        ratio = width/height
        
        ## Get pixel position untransformed
        pixelVecX = ((pixelX/width) * (ratio)) - ((ratio)/2)
        pixelVecY = 0.5 - (pixelY/height)
        
        ## Transform position
        pixelVec = Vec3(pixelVecX, pixelVecY, camera.length/20).applyTransforms(camera.position, Scale(1, 1, 1), camera.rotation)

        return pixelVec
    
    ## Calculate initial ray
    def getRay(self, pixelVec) -> tuple[Vec3, Vec3]:
        scene = self.transformedScene if self.options['transformBefore'] else self.scene

        rayStart = scene.camera.position
        rayDirection = (pixelVec - rayStart).normalise()
        
        return Ray(rayStart, rayDirection, 1, None)

    ## Use BVH structure to find all of the relevant faces for a collision
    def getFaces(self, ray, bvh):
        ## Base cases
        if not ray.entersAABB(bvh.box):
            return []
        if bvh.faces:
            return bvh.faces

        leftFaces = None
        rightFaces = None

        if bvh.left:
            leftFaces = self.getFaces(ray, bvh.left)
        if bvh.right:
            rightFaces = self.getFaces(ray, bvh.right)

        return leftFaces + rightFaces
    
    ## Return the collision between a ray and a sphere (if it exists)
    def getSphereCollision(self, ray, object, backCull):
        ## Setup sphere object based on properties
        sphere = MathematicalSphere(object.position, object.scale.x/2)
        
        ## Get front and back collision distances
        ts = ray.getTSphere(sphere)
        if ts != (None, None):            
            ## If collisions are behind ray start point
            if ts[0] > 0:
                t = ts[0]
                direction = 0
            elif ts[1] > 0 and not backCull and object.type != 'light':
                t = ts[1]
                direction = 1
            else:
                return None

            point = ray.pointOnRay(t)
            
            normal = (point-sphere.position).normalise()
            
            return {
                'point': point,
                't': t,
                'normal': normal,
                'direction': direction,
            }

    ## Check if a given point is within a triangle of 3 points
    def pointWithinTriangle(self, point, vertices):
        ## Vectors between vertices anticlockwise
        vectorAtoB = vertices[1]-vertices[0]
        vectorBtoC = vertices[2]-vertices[1]
        vectorCtoA = vertices[0]-vertices[2]
        
        ## Vectors between vertices and collision point
        vectorAtoPoint = point - vertices[0]
        vectorBtoPoint = point - vertices[1]
        vectorCtoPoint = point - vertices[2]
        
        ## Cross products
        crossA = vectorAtoB.cross(vectorAtoPoint)
        crossB = vectorBtoC.cross(vectorBtoPoint)
        crossC = vectorCtoA.cross(vectorCtoPoint)

        ## If all cross products face the same direction
        if (crossA.dot(crossB) > 0) and (crossB.dot(crossC) > 0):
            return True
        
        return False

    ## Return the collision between a ray and a face (if it exists)
    def getFaceCollision(self, ray, vertices):
        ## Get face plane
        facePlane = MathematicalPlane(vertices)
        
        if ray.direction.perpendicular(facePlane.normal):
            return None
        else:
            ## Get collision distance
            t = ray.getTPlane(facePlane)
            
            ## If collision is behind ray start
            if t < 0:
                return None
            
            point = ray.pointOnRay(t)
            
            if self.pointWithinTriangle(point, vertices):
                if ray.direction.dot(facePlane.normal) < 0:
                    direction = 0
                else:
                    direction = 1
                
                return {
                    'point': point,
                    't': t,
                    'normal': facePlane.normal,
                    'direction': direction,
                }

    ## Find the closest collision of a ray with an object
    def getFirstCollision(self, ray, lights, backCull=True):
        ## Default values to None
        collisionObj = None
        collisionPoint = None
        collisionT = None
        collisionNormal = None
        collisionFace = None
        collisionDirection = None
        
        scene = self.transformedScene if self.options['transformBefore'] else self.scene

        for object in scene.objects:
            if object.type not in ['sphere', 'mesh', 'light']:
                continue

            ## Check if ray enters object AABB
            if (self.options['aabb'] or self.options['bvh']):
                if not self.options['transformBefore']:
                    object.bvh = object.buildBVH()
                
                if not ray.entersAABB(object.bvh.box):
                    continue
            
            ## If object is sphere-based
            if object.type == 'sphere' or (lights and object.type == 'light'):
                collisionInfo = self.getSphereCollision(ray, object, backCull)
                
                ## If it is the first or closest collision
                if collisionInfo and (collisionObj == None or collisionInfo['t'] < collisionT):
                    collisionObj = object
                    collisionPoint = collisionInfo['point']
                    collisionT = collisionInfo['t']
                    collisionNormal = collisionInfo['normal']
                    collisionDirection = collisionInfo['direction']

            ## If object is mesh-based
            elif object.type == 'mesh':
                ## Get relevant faces
                if self.options['bvh']:
                    faces = self.getFaces(ray, object.bvh)
                else:
                    faces = object.mesh.faces

                for face in  faces:
                    faceVerticesLocal = object.mesh.faceVertices(face)
                    if not self.options['transformBefore']:
                        ## Transform face vertices
                        faceVertices = []
                        
                        for vertex in faceVerticesLocal:
                            vertexAbsolute = vertex.applyTransforms(object.position, object.scale, object.rotation)
                            faceVertices.append(vertexAbsolute)
                    else:
                        ## Get already-transformed faces
                        faceVertices = faceVerticesLocal
                    
                    collisionInfo = self.getFaceCollision(ray, faceVertices)

                    ## Ignore back faces if backCull is True
                    if collisionInfo and (collisionInfo['direction'] == 0 or not backCull):
                        ## If it is the first or closest collision
                        if collisionObj == None or collisionInfo['t'] < collisionT:
                            collisionObj = object
                            collisionPoint = collisionInfo['point']
                            collisionT = collisionInfo['t']
                            collisionNormal = collisionInfo['normal']
                            collisionDirection = collisionInfo['direction']
                            collisionFace = face

        return {
            'object': collisionObj,
            'point': collisionPoint,
            't': collisionT,
            'normal': collisionNormal,
            'direction': collisionDirection,
            'face': collisionFace,
        }

    ## Get the ray after bouncing off a surface
    def bounce(self, ray, collisionInfo) -> Ray:
        ## Define information
        bouncePoint = collisionInfo['point']
        obj = collisionInfo['object']
        normal = collisionInfo['normal'].normalise()

        ## Get a perfectly diffusive reflection; using multiple samples accounts for the randomness
        diffuseDirection = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
        diffuseDirection = diffuseDirection.normalise()
        if diffuseDirection.dot(normal) < 0:
            diffuseDirection = diffuseDirection * -1
        
        ## Get a perfectly specular reflection
        specularDirection = ray.direction.flip(normal)
        
        ## Get roughness of the face
        face = collisionInfo['face']

        shader = obj.getShader(face)
        roughness = shader.roughness if shader else 1
        
        ## Combine diffusive and specular reflections to calculate ray
        bounceDirection = (specularDirection * (1-roughness)) + (diffuseDirection * roughness)
        bounceStart = bouncePoint + bounceDirection*0.1
        returnRay = Ray(bounceStart, bounceDirection, ray.ior, ray.object)

        return returnRay

    ## Get the visibility of a ray to a light source to calculate shadows
    def getVisibility(self, ray) -> float:
        ## Get closest collision
        collisionInfo = self.getFirstCollision(ray, True, False)
        collisionObj = collisionInfo['object']
        collisionFace = collisionInfo['face']
        
        ## Check if the closest collision(?) is not a light or an emissive surface, i.e. the light is blocked by an object
        if (collisionObj != None
            and (collisionObj.type != 'light'
                 and not (collisionObj.type == 'mesh'
                          and collisionObj.getShader(collisionFace).emissionStrength > 0))):
            ## Calculate visbility based on how opaque an object is
            visibility = 1-collisionObj.getShader(collisionFace).reflectivity
            return visibility
        else:
            ## Completely exposed to light source
            return 1

    # MARK: > ATTENUATION
    ## Calculate the attenuation (falloff) of a light over distance
    def getAttenuation(self, distance, k=0.01) -> float:
        attenuation = 1 / (1 + k * distance ** 2)

        return max(0, min(1, attenuation))
    
    # MARK: > GET LIGHTS POS
    ## Get the position of all light sources in the scene
    def getAllLightsPos(self):
        lights = []

        scene = self.transformedScene if self.options['transformBefore'] else self.scene

        for object in scene.objects:
            ## If object emits light
            if object.type == 'light' or (object.type == 'mesh' and any(shader.emissionStrength > 0 for shader in object.shaders)):
                ## If object is emissive mesh
                if object.type == 'mesh':
                    for shader in object.shaders:
                        if shader.emissionStrength > 0:
                            for faceIndex, face in enumerate(object.mesh.faces):
                                if object.getShader(face) == shader:
                                    if self.options['debug']:
                                        print('face:', faceIndex)
                                    strength = shader.emissionStrength
                                    color = shader.color
                                    pos = Mesh.centroid(*[object.mesh.vertices[index] for index in face])
                                    lights.append([pos, strength, color])
                ## If object is a light
                else:
                    pos = object.position
                    strength = object.strength
                    color = object.shaders[0].color
                    lights.append([pos, strength, color])
    
        return lights
    
    # MARK: > DIRECT LIGHT
    ## Calculate the direct light intensity at a collision point
    def getDirectLight(self, collisionInfo) -> float | int:
        totalColor = RGB(0, 0, 0)
        totalIntensity = 0
        bouncePoint = collisionInfo['point']
        
        lights = self.getAllLightsPos()

        scene = self.transformedScene if self.options['transformBefore'] else self.scene

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
            viewVector = (scene.camera.position - collisionInfo['point'])
            
            ## Get intensity of light on a perfectly specular material
            specularIntensity = max((0, (0 if reflectionVector.normalise().dot(viewVector.normalise()) < 0.96 else 1)))
            # if collisionInfo['object'].name == 'Mirror Ball':
            #     print('specularIntensity:', specularIntensity)
            
            ## Get roughness of material at collision point
            # roughness = collisionInfo['object'].shader.roughness
            # roughness = 1
            # if collisionInfo['object'].type == 'mesh':
            #     roughness = collisionInfo['object'].shaders[collisionInfo['object'].shaderIndices[collisionInfo['object'].mesh.faces.index(collisionInfo['face'])]].roughness
            # elif collisionInfo['object'].type in ['empty', 'sphere']:
            #     roughness = collisionInfo['object'].shaders[0].roughness
            shader = collisionInfo['object'].getShader(collisionInfo['face'])
            roughness = shader.roughness if shader else 1
            # roughness = collisionInfo['object'].shaders[collisionInfo['object'].mesh.faces.index(collisionInfo['face'])].roughness
            
            ## Get visibility to determine shadows
            visibility = self.getVisibility(bounceRay)
            
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
        ## Get information for refraction
        normal = collisionInfo['normal'].normalise() ## N
        direction = ray.direction.normalise() ## I
        incidentObject = ray.object
        incidentIOR = ray.ior ## n1
        transmittedObject = collisionInfo['object']
        transmittedIOR = transmittedObject.material.ior ## n2
        
        ## Flip information if exiting an object instead of entering
        if transmittedObject == incidentObject:
            normal = normal * -1
            transmittedIOR = 1
            transmittedObject = None
        
        ## Snell's law maths
        cosIncident = direction.dot(normal) * -1 ## cosθI
        sinIncidentSquared = 1 - (cosIncident**2) ## sin2θI
        
        ratio = incidentIOR / transmittedIOR ## ŋ
        
        sinTransmittedSquared = ratio**2 * sinIncidentSquared ## sin2θR
        if sinTransmittedSquared > 1:
            return None
        cosTransmitted = math.sqrt(1 - sinTransmittedSquared) ## cosθR
        
        ## Vector derivation of snell's law
        parallelComponent = direction * ratio ## ŋI
        perpendicularComponent = normal * ((ratio * cosIncident) - cosTransmitted) ## (ŋcosθI - cosθR)N

        specularDirection = parallelComponent + perpendicularComponent ## R

        ## Get a perfectly diffusive refraction; using multiple samples accounts for the randomness
        diffuseDirection = Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
        diffuseDirection = diffuseDirection.normalise()
        if diffuseDirection.dot(normal) > 0:
            diffuseDirection = diffuseDirection * -1
        
        ## Get roughness of the face
        face = collisionInfo['face']
        shader = collisionInfo['object'].getShader(face)
        roughness = shader.roughness if shader else 1
        
        ## Combine diffusive and specular refractions to calculate ray
        bounceDirection = (specularDirection * (1-roughness)) + (diffuseDirection * roughness)
        bounceStart = collisionInfo['point'] + bounceDirection*0.1

        # refractedRay = Ray(bounceStart, bounceDirection, ray.ior, ray.object)
        refractedRay = Ray(bounceStart, bounceDirection, transmittedIOR, transmittedObject)
        
        # refractedRay = Ray((collisionInfo['point'] + (refractedDirection * 0.1)), refractedDirection, transmittedIOR, transmittedObject)
        
        return refractedRay

    ## Trace a ray's path until it reaches a stopping condition
    def trace(self, ray, bounces, lights=False) -> TraceTreeNode | None:
        ## Get collision and node info
        collisionInfo = self.getFirstCollision(ray, lights, ray.ior == 1)
        node = TraceTreeNode(ray, collisionInfo)
        object = collisionInfo['object']
        face = collisionInfo['face']
        # print(collisionInfo)
        
        if object:
            shader = object.getShader(face)
        
        ## Base cases
        if collisionInfo['point'] == None:
            return None
        elif lights and (object.type == 'light'
                         or (object.type == 'mesh' and shader.emissionStrength > 0)):
            return node
        elif bounces == 0:
            return node
        
        reflectivity = shader.reflectivity if shader else 1
        
        transmitNode = None
        ## If somewhat transparent
        if reflectivity != 1:
            ## Calculate transmitted ray
            transmitRay = self.transmit(ray, collisionInfo)
            if transmitRay != None:
                transmitNode = self.trace(transmitRay, bounces-1, lights=True)

        bounceNode = None
        ## If somewhat opaque
        if reflectivity != 0:
            ## Calculate reflected ray
            bounceRay = self.bounce(ray, collisionInfo)
            if bounceRay != None:
                bounceNode = self.trace(bounceRay, bounces-1, lights=True)
        
        ## Chain nodes to create trace tree
        node.left = bounceNode
        node.right = transmitNode

        return node

    ## (Recursive) Compute the color of a pixel given its trace
    def computeColor(self, trace) -> RGB:
        ## Base case
        if trace == None:
            return self.options['ambient']
        
        obj = trace.collisionInfo['object']
        
        scene = self.transformedScene if self.options['transformBefore'] else self.scene
        
        ## Object is light
        if obj.type == 'light':
            ## Get light color and strength
            # color = obj.shader.color
            color = obj.shaders[0].color
            color *= obj.strength
            
            ## Multiply by attenuation
            distance = (trace.collisionInfo['point'] - scene.camera.position).magnitude()
            directAttenuation = self.getAttenuation(distance)
            color *= directAttenuation
            
            return color.clamp()

        ## Object is emissive mesh
        if obj.type == 'mesh' and obj.shaders[obj.shaderIndices[obj.mesh.faces.index(trace.collisionInfo['face'])]].emissionStrength > 0:
            ## Get face color and strength
            # color = obj.shader.color
            shader = obj.getShader(trace.collisionInfo['face'])
            color = shader.color * shader.emissionStrength
            # color = obj.shaders[obj.shaderIndices[obj.mesh.faces.index(trace.collisionInfo['face'])]].color
            # color *= obj.shaders[obj.shaderIndices[obj.mesh.faces.index(trace.collisionInfo['face'])]].emissionStrength
            
            ## Multiply by attenuation
            distance = (trace.collisionInfo['point'] - scene.camera.position).magnitude()
            directAttenuation = self.getAttenuation(distance)
            color *= directAttenuation
            
            return color.clamp()
        
        ## Get shader from object
        # shader = obj.shader
        # if obj.type == 'mesh':
        #     shader = obj.shaders[obj.shaderIndices[obj.mesh.faces.index(trace.collisionInfo['face'])]]
        # elif obj.type in ['empty', 'camera', 'sphere']:
        #     shader = obj.shaders[0]
        shader = obj.getShader(trace.collisionInfo['face'])
        
        color = shader.color

        ## Recursion to get colors from next rays
        reflectedColor = self.computeColor(trace.left) if trace.left is not None else self.options['ambient'] * (2/255)
        transmittedColor = self.computeColor(trace.right) if trace.right is not None else self.options['ambient'] * (2/255)
        
        ## Factors for combination
        reflectivity = shader.reflectivity
        transmissionFactor = 1 - reflectivity
        
        ## Get attenuations
        reflectedDistance = (trace.left.collisionInfo['point']-trace.ray.start).magnitude() if trace.left is not None else 0
        reflectedAttenuation = self.getAttenuation(reflectedDistance)        
        transmittedDistance = (trace.right.collisionInfo['point']-trace.ray.start).magnitude() if trace.right is not None else 0
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

    ## Create a Rendered object based on the scene properties and data
    def render(self, progressCallback=lambda _, __, ___: None, cancelCallback=lambda: False, returnType='rendered') -> Rendered | np.ndarray:
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

        if self.options['height'] % self.options['threads'] != 0:
            raise ValueError('Number of threads must be a factor of image height')
        
        stripeHeight = self.options['height'] // self.options['threads']

        ## Store information for Rendered object later
        totalTraces = {}
        self.rays = 0
        start = time.time()

        renderQueue = Queue()
        threads = []
        
        self.threadsProgress = ThreadsProgress(callback=progressCallback, total=stripeHeight*self.options['width'])

        ## Start all render threads
        for threadId in range(self.options['threads']):
            thread = threading.Thread(target=self.renderStripe, args=(stripeHeight, cancelCallback, threadId, renderQueue))
            threads.append(thread)
            thread.start()
        
        if self.options['debug']:
            print('started all threads')
        
        ## Wait for threads to finish and join them back up
        for thread in threads:
            thread.join()
        
        if self.options['debug']:
            print('finished rendering')
        
        ## Get the results from the queue and put them into a list
        renderResults = []
        while not renderQueue.empty():
            renderResult = renderQueue.get()
            renderResults.append(renderResult)
        
        ## Sort them by id
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
        
        print('merging')
        # image = self.mergeImgs(images, self.options['width'], self.options['height'])
        image = np.vstack(stripes)

        print('finished merging')

        ## Set final Rendered properties and then return
        totalTime = end-start
        rays = self.rays
        self.rays = 0
        
        print(totalTime)
        
        print('returning')
        
        if returnType == 'rendered':
            return Rendered(image, totalTraces, self.options['width'], self.options['height'], totalTime, rays, self.options, failed=failed, engine='photon')
        elif returnType == 'np':
            return image

    def renderStripe(self, stripeHeight, cancelCallback, id=None, queue=None):
        self.threadsProgress[id] = 0
        if self.options['debug']:
            print(f'Thread {id} started')
        ## Setup image
        # image = np.zeros((self.options['height'], self.options['width'], 3), np.uint8)
        stripe = np.zeros((stripeHeight, self.options['width'], 3), np.uint8)
        
        totalTraces = {}
        
        failed = False
        
        startTime = time.time()

        startRow = id*stripeHeight

        for pixelY in range(stripeHeight):
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
                    color = RGB.mean(samples).toTuple('bgr')
                    
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
        if self.options['debug']:
            print(f'Thread {id} done in {timeToComplete:.2f}')
        queue.put([id, stripe, totalTraces, failed])


# MARK: THREADS PROGRESS
## Structure to allow tracking of threads progress
class ThreadsProgress:
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
        defaultOptions = {'debug': False, 'selectedFaces': [], 'attenuation': True, 'backCull': True, 'vertices': False, 'lights': False, 'edges': False, 'axes': False, 'selectedObject': -1, 'normals': False, 'ambient': RGB(0, 0, 0), 'width': 1440, 'height': 900, 'lighting': False}
        for optionKey in defaultOptions.keys():
            if optionKey not in options.keys():
                options[optionKey] = defaultOptions[optionKey]
        
        return options

    # MARK: > PROJECT
    ## Get vertex position on screen
    def projectVertex(self, vertex, cullBehind=True):
        camera = self.transformedScene.camera
        
        ## Vertex's object transforms are already applied
        ## Apply camera position so that the camera is effectively at (0, 0, 0)
        transformedVertexNoRotation = vertex.applyTransforms(camera.position * -1, Scale(1, 1, 1), Euler(0, 0, 0))
        
        transformedVertex = transformedVertexNoRotation.applyTransforms(Vec3(0, 0, 0), Scale(1, 1, 1), camera.rotation * -1)
        
        ## Check vertex is not behind camera
        if transformedVertex.z < 0 and cullBehind:
            return None

        if transformedVertex.z == 0:
            return None
        
        ## Project the vertex onto the screen space
        scaleFac = (camera.length/20) / transformedVertex.z
        vecX = transformedVertex.x * scaleFac
        vecY = transformedVertex.y * scaleFac
        
        ## Convert vertex vector into pixel coordinates using screen width
        halfWidth = (self.options['width']/self.options['height']) / 2
        pixelX = ((halfWidth + vecX)/(self.options['width']/self.options['height'])) * self.options['width']
        pixelY = (0.5 - vecY) * self.options['height']
        
        if pixelX < 0 or pixelY < 0:
            return None
        
        return (int(pixelX), int(pixelY))

    def renderAxes(self, image, clipOrigin=Vec3(0,0,0)):
        for pos in range(int(clipOrigin.x-16), int(clipOrigin.x+16)):
            negativePix = self.projectVertex(Vec3(pos, 0, clipOrigin.z-16), cullBehind=True)
            positivePix = self.projectVertex(Vec3(pos, 0, clipOrigin.z+16), cullBehind=True)
            print(negativePix, positivePix)
            
            if positivePix and negativePix:
                color = (120, 120, 120)
                if pos == 0:
                        color = (180, 0, 0)
                
                cv2.line(image, negativePix, positivePix, color, 1)

        for pos in range(int(clipOrigin.z-16), int(clipOrigin.z+16)):
            negativePix = self.projectVertex(Vec3(clipOrigin.x-16, 0, pos), cullBehind=True)
            positivePix = self.projectVertex(Vec3(clipOrigin.x+16, 0, pos), cullBehind=True)
            
            if positivePix and negativePix:
                color = (120, 120, 120)
                if pos == 0:
                        color = (0, 0, 180)
                
                cv2.line(image, negativePix, positivePix, color, 1)
        
        negativeYPix = self.projectVertex(Vec3(0, -8, 0), cullBehind=True)
        positiveYPix = self.projectVertex(Vec3(0, 8, 0), cullBehind=True)
        
        if positiveYPix and negativeYPix:
            cv2.line(image, negativeYPix, positiveYPix, (0, 180, 0), 1)

    def getDrawables(self, objects, camera):
        drawables = []
        
        for obj in objects:
            if obj.type in ['sphere', 'light']:
                drawables.append({
                    'type': obj.type,
                    'id': obj.id,
                    'centroid': obj.position,
                    'radius': obj.scale.x/2,
                })
            elif obj.type == 'mesh':
                for face in obj.mesh.faces:
                    vertices = [obj.mesh.vertices[vertexID] for vertexID in face]
                    drawables.append({
                        'type': 'face',
                        'id': obj.id,
                        'centroid': Mesh.centroid(*vertices),
                        'face': face,
                        'vertices': vertices,
                    })
        
        drawables.sort(key=lambda drawable: (drawable['centroid']-camera.position).magnitude(), reverse=True)
        
        return drawables

    ## Render a simple rasterised image of a scene with projection
    def render(self):
        if self.options['debug']:
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
        if self.options['attenuation'] or self.options['backcull'] or self.options['lighting']:
            photon = Photon(scene=self.transformedScene, options={'transformBefore': False})

        ## Setup image to work on
        # image = np.zeros((self.options['height'], self.options['width'], 3), np.uint8)
        image = np.full((self.options['height'], self.options['width'], 3), self.options['ambient'].toTuple('bgr'), dtype=np.uint8)
        
        ## Start timing render
        start = time.time()

        if self.options['axes']:
            self.renderAxes(image, self.transformedScene.camera.position)

        drawables = self.getDrawables(self.transformedScene.objects, self.transformedScene.camera)

        for drawable in drawables:
            if drawable['type'] == 'sphere':
                projectedPoint = self.projectVertex(drawable['centroid'])
                projectedCircumferencePoint = self.projectVertex(drawable['centroid'] + Vec3(drawable['radius'], 0, 0))
            
                
                if projectedPoint and projectedCircumferencePoint:
                    dx = abs(projectedCircumferencePoint[0] - projectedPoint[0])
                    dy = abs(projectedCircumferencePoint[1] - projectedPoint[1])
                    screenspaceRadius = round(math.sqrt(dx**2 + dy**2))
                    
                    color = self.transformedScene.getObject(drawable['id']).shaders[0].color
                    shadedColor = color
                    if self.options['lighting']:
                        ## Get closest light source
                        lights = photon.getAllLightsPos()
                        for light in lights:
                            dist = (light[0] - drawable['centroid']).magnitude()
                            attenuation = photon.getAttenuation(dist)
                            strength = light[1]
                            color = light[2]
                            ## Get shaded color
                            shadedColor = (shadedColor * (self.options['ambient']/255) * attenuation * (strength / 100)) * color
                        
                    if self.options['attenuation']:
                        ## Get attenuation
                        dist = (drawable['centroid'] - self.transformedScene.camera.position).magnitude()
                        attenuation = photon.getAttenuation(dist)
                        ## Get shaded color
                        shadedColor = shadedColor * attenuation
                    
                    cv2.circle(image, projectedPoint, screenspaceRadius, shadedColor.toTuple('bgr'), -1)

                    if self.options['edges']:
                        ## Draw circumference
                        lineColor = (0, 0, 0)
                        ## Make selected sphere have orange circumference
                        if drawable['id'] == self.options['selectedObject']:
                            lineColor = (0, 100, 255)
                        cv2.circle(image, projectedPoint, screenspaceRadius, lineColor, 1)
            
            elif drawable['type'] == 'face':
                projected = [self.projectVertex(vertex) for vertex in drawable['vertices']]
                obj = self.transformedScene.getObject(drawable['id'])

                if all([pix != None for pix in projected]):
                    facePlane = MathematicalPlane(drawable['vertices'])
                    if facePlane.normal.dot((self.transformedScene.camera.position - drawable['centroid'])) < 0:
                        normalDirection = 'away'
                    else:
                        normalDirection = 'toward'

                    ## Cull any faces facing away
                    if self.options['backCull'] and normalDirection == 'away':
                        continue
                    
                    ## Display object color
                    shader = obj.getShader(drawable['face'])
                    color = shader.color
                    # color = obj.shaders[obj.shaderIndices[obj.mesh.faces.index(face)]].color
                    
                    color += self.options['ambient']

                    if self.options['normals']:
                        ## Display red/blue depending on normal direction
                        if normalDirection == 'away':
                            color = RGB(230, 0, 0)
                        elif normalDirection == 'toward':
                            color = RGB(0, 0, 230)

                    elif obj.mesh.faces.index(drawable['face']) in self.options['selectedFaces']:
                        color = RGB(255, 228, 157)

                    shadedColor = color
                    if self.options['lighting']:
                        ## Get closest light source
                        lights = photon.getAllLightsPos()
                        for light in lights:
                            dist = (light[0] - obj.position).magnitude()
                            attenuation = photon.getAttenuation(dist)
                            strength = light[1]
                            color = light[2]
                            ## Get shaded color
                            shadedColor = shadedColor = (shadedColor * (self.options['ambient']/255) * attenuation * (strength / 20)) * color

                    if self.options['attenuation']:
                        ## Get attenuation
                        dist = (obj.position - self.transformedScene.camera.position).magnitude()
                        attenuation = photon.getAttenuation(dist)
                        ## Get shaded color
                        shadedColor = shadedColor * attenuation
                    
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
                            cv2.circle(image, vertex, 3, color=vertexColor, thickness=-1)
        
            elif drawable['type'] == 'light' and self.options['lights']:
                projectedPoint = self.projectVertex(drawable['centroid'])
                
                ## Calculate a screenspace radius by projecting a circumference point
                projectedCircumferencePoint = self.projectVertex(drawable['centroid'] + Vec3(drawable['radius'], 0, 0))
                
                if projectedPoint and projectedCircumferencePoint:
                    dx = abs(projectedCircumferencePoint[0] - projectedPoint[0])
                    dy = abs(projectedCircumferencePoint[1] - projectedPoint[1])
                    screenspaceRadius = round(math.sqrt(dx**2 + dy**2))
                    
                    color = (0, 0, 0)
                    ## Make selected light have orange circumference
                    if drawable['id'] == self.options['selectedObject']:
                        color = (0, 100, 255)
                    
                    cv2.circle(image, projectedPoint, screenspaceRadius, color=color, thickness=1)

        if self.options['selectedObject'] == self.transformedScene.camera.id and self.options['edges']:
            cv2.rectangle(image, (0, 0), (self.options['width']-1, self.options['height']-1), (0, 100, 255), 3)
            
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