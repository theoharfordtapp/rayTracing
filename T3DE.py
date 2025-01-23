# AQA A-Level Computer Science NEA 2024
#
# Graphics Engine
# Theo HT
#
# (3D Engine)

import trimesh
import random
import math
import json
import copy


## 9x9 matrix used to calculate rotations
class Matrix9:
    def __init__(self, elements):
        self.elements = elements
    
    def __mul__(self, other):
        ## Multiply by vector
        if type(other) == Vec3:
            e = self.elements
            return Vec3(
                e[0] * other.x + e[1] * other.y + e[2] * other.z,
                e[3] * other.x + e[4] * other.y + e[5] * other.z,
                e[6] * other.x + e[7] * other.y + e[8] * other.z
            )
        ## Multiply by another matrix
        elif type(other) == Matrix9:
            e = self.elements
            o = other.elements
            return Matrix9([
                e[0]*o[0] + e[1]*o[3] + e[2]*o[6],
                    e[0]*o[1] + e[1]*o[4] + e[2]*o[7],
                        e[0]*o[2] + e[1]*o[5] + e[2]*o[8],
                e[3]*o[0] + e[4]*o[3] + e[5]*o[6],
                    e[3]*o[1] + e[4]*o[4] + e[5]*o[7],
                        e[3]*o[2] + e[4]*o[5] + e[5]*o[8],
                e[6]*o[0] + e[7]*o[3] + e[8]*o[6],
                    e[6]*o[1] + e[7]*o[4] + e[8]*o[7],
                        e[6]*o[2] + e[7]*o[5] + e[8]*o[8]
            ])
        else:
            raise TypeError("Matrix9 can only be multiplied by Vec3 or another Matrix9")


## Euler angle used to save rotation information
class Euler:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other):
        ## Add to other Euler
        if type(other) == Euler:
            return Euler(self.x + other.x, self.y + other.y, self.z + other.z)
        else:
            raise TypeError(f'Cannot add type {type(other)} to Euler')

    def __mul__(self, other):
        ## Multiply by number
        if type(other) in [float, int]:
            return Euler(self.x * other, self.y * other, self.z * other)
        else:
            raise TypeError(f'Cannot multiply Euler with type {type(other)}')

    ## Convert to matrix
    def toMatrix(self):
        ## Get trig values
        cosX, sinX = math.cos(self.x), math.sin(self.x)
        cosY, sinY = math.cos(self.y), math.sin(self.y)
        cosZ, sinZ = math.cos(self.z), math.sin(self.z)

        ## Create matrices
        Rx = Matrix9([
            1, 0, 0,
            0, cosX, -sinX,
            0, sinX, cosX
        ])
        Ry = Matrix9([
            cosY, 0, sinY,
            0, 1, 0,
            -sinY, 0, cosY
        ])
        Rz = Matrix9([
            cosZ, -sinZ, 0,
            sinZ, cosZ, 0,
            0, 0, 1
        ])

        ## Combine matrices
        return Rz * Ry * Rx

    ## Use euler as an iterable
    def __iter__(self) -> iter:
        return iter([self.x, self.y, self.z])
    
    def __getitem__(self, index) -> int:
        return [self.x, self.y, self.z][index]
    
    def __setitem__(self, index, val) -> int:
        match index:
            case 0:
                self.x = val
            case 1:
                self.y = val
            case 2:
                self.z = val

    def __repr__(self) -> str:
        return f'◿ {self.x}, {self.y}, {self.z} ◺'
    
    def __format__(self, formatSpec: str) -> str:
        ## Format each component of the vector based on the format specifier
        formattedX = format(self.x, formatSpec)
        formattedY = format(self.y, formatSpec)
        formattedZ = format(self.z, formatSpec)
        
        return f'◿ {formattedX}, {formattedY}, {formattedZ} ◺'


## Ray used for tracing paths
class Ray:
    def __init__(self, start, direction, ior, object) -> None:
        self.start = start
        self.direction = direction.normalise()
        self.ior = ior
        self.object = object

    ## Get distance along ray where a collision with a plane occurs
    def getTPlane(self, plane):
        ## Get ray and plane information
        start = self.start
        direction = self.direction
        normal = plane.normal
        origin = plane.position
        
        ## Equation to find the point on the ray which is also on the plane
        t = (normal.dot(origin) - normal.dot(start))/(normal.dot(direction))
        return t

    ## Get distance along ray where a collision with a sphere occurs
    def getTSphere(self, sphere):
        ## Get ray and sphere information
        center = sphere.position ## C
        relativeStart = self.start - center ## L
        direction = self.direction ## D
        radius = sphere.radius ## r
        
        ## Find discriminant to check how many collision points
        discriminant = (relativeStart.dot(direction)**2) - (relativeStart.dot(relativeStart)-(radius**2))
        
        ## Ray misses sphere
        if discriminant < 0: return (None, None)

        ## Get the distances along the ray (t) if they exist. t1 is the first solution, t2 the second.
        ## If there is only one collision, there are still 2 solutions, but both are identical.
        t1 = (relativeStart.dot(direction) * -1) - math.sqrt(discriminant)
        t2 = (relativeStart.dot(direction) * -1) + math.sqrt(discriminant)

        return (t1, t2)

    ## Check if a ray ever enters a bounding box
    def entersAABB(self, box):
        t = [0, 0, 0]
        
        ## Iterate through all 3 axes
        for axis in range(3):
            ## Get the component of the origin and direction vectors along this axis
            originComponent = self.start[axis]
            directionComponent = self.direction[axis]
            
            ## Ray is perpendicular to this axis
            if directionComponent == 0:
                if originComponent < box[0][axis] or originComponent > box[1][axis]:
                    ## If the ray is perpendicular to this axis and starts outside the box on this axis
                    return False
                
                ## If the ray is perpendicular but starts within the box on this axis
                t[axis] = (float('-inf'), float('inf'))
                continue
            
            ## Get the component of the box's min and max coordinates on this axis
            boxMinComponent = box[0][axis]
            boxMaxComponent = box[1][axis]
            
            ## Get the t value where the ray intersects the box's walls on this axis
            tEnterAxis = (boxMinComponent - originComponent)/directionComponent
            tExitAxis = (boxMaxComponent - originComponent)/directionComponent
            
            ## Swap if the ray is going the other direction
            if tEnterAxis > tExitAxis:
                tEnterAxis, tExitAxis = tExitAxis, tEnterAxis
            
            t[axis] = (tEnterAxis, tExitAxis)
        
        ## Calculate where the ray enters and exits the box
        tEnter = max([axis[0] for axis in t])
        tExit = min([axis[1] for axis in t])
        
        ## Return if the ray enters the box before it exists and if it exits past the ray start point
        return tEnter <= tExit and tExit > 0

    ## Convert t distance into absolute vector
    def pointOnRay(self, t):
        return self.start + (self.direction*t)

    ## Convert ray to normalised vector
    def toVec3(self):
        return (self.pointOnRay(1) - self.start).normalise()

    ## Convert ray to a dictionary
    def toDict(self):
        ## Convert the current object to a dictionary
        objectDict = self.object.toDict() if self.object else None
        
        ## Combine object dict with other information from ray
        return {
            'start': [*self.start],
            'direction': [*self.direction],
            'ior': self.ior,
            'object': objectDict,
        }

    ## Convert dictionary to a ray
    @staticmethod
    def fromDict(data):
        ## Convert the object data back into an object
        object = Object.fromDict(data.get('object', None)) if data.get('object', None) else None
        
        ## Combine object with other information from ray data
        return Ray(
            start=Vec3(*data.get('start', [0, 0, 0])),
            direction=Vec3(*data.get('direction', [0, 0, 0])),
            ior=data.get('ior', 1),
            object=object,
        )
    
    def __repr__(self) -> str:
        return f'<{self.start}, {self.direction}>'
    
    def __format__(self, formatSpec) -> str:
        formattedStart = format(self.start, formatSpec)
        formattedDirection = format(self.direction, formatSpec)
        return f'<{formattedStart}, {formattedDirection}>'


## Mathematically-represented plane object, instead of mesh
class MathematicalPlane:
    def __init__(self, vertices) -> None:
        self.position = vertices[0]
        self.normal = self._normal(vertices)
    
    def _normal(self, vertices):
        firstVec = vertices[1] - vertices[0]
        secondVec = vertices[2] - vertices[0]
        
        ## Cross vectors to get plane normal
        normalVec = firstVec.cross(secondVec)
        
        return normalVec


## Mathematically-represented sphere object, instead of mesh approximation
class MathematicalSphere:
    def __init__(self, position, radius) -> None:
        self.position = position
        self.radius = radius


## Node of a ray trace tree, used to store information about a ray trace
class TraceTreeNode:
    def __init__(self, ray, collisionInfo, left=None, right=None) -> None:
        self.ray = ray
        self.collisionInfo = collisionInfo
        self.left = left if left is not None else None
        self.right = right if right is not None else None

    ## Convert trace tree node to a dictionary
    def toDict(self):
        ## Get the dictionary representations of the left and right node
        leftNode = self.left.toDict() if self.left else None
        rightNode = self.right.toDict() if self.right else None
        
        ## Get the face in integers (rather than np.int64 types)
        face = [int(vertex) for vertex in [*self.collisionInfo['face']]] if self.collisionInfo['face'] != None else None
        
        ## Get the dictionary representation of the collisionInfo
        collisionInfo = {
            'coordinate': [*self.collisionInfo['coordinate']],
            'object': self.collisionInfo['object'].toDict(),
            'normal': [*self.collisionInfo['normal']],
            'direction': int(self.collisionInfo['direction']),
            'face': face,
        }
        
        ## Combine the information
        return {
            'ray': self.ray.toDict(),
            'collisionInfo': collisionInfo,
            'left': leftNode,
            'right': rightNode,
        }

    ## Convert dictionary to a trace tree node
    @staticmethod
    def fromDict(data):
        ## Get the left and right node from dictionary data
        leftNode = TraceTreeNode.fromDict(data.get('left', None)) if data.get('left', None) else None
        rightNode = TraceTreeNode.fromDict(data.get('right', None)) if data.get('right', None) else None
        
        ## Set default ray data in case a ray is not found
        defaultRayData = {
            'start': [0, 0, 0],
            'direction': [0, 0, 0],
        }
        ## Convert ray data to ray
        ray = Ray.fromDict(data.get('ray', defaultRayData))
        
        ## Set default collisionInfo data in case the data is not found
        defaultCollisionInfoData = {
            'coordinate': None,
            'object': None,
            'normal': None,
            'direction': None,
            'face': None,
        }
        ## Get collisionInfo data
        collisionInfoData = data.get('collisionInfo', defaultCollisionInfoData)
        face = tuple(collisionInfoData['face']) if collisionInfoData['face'] != None else None
        
        ## Convert collisionInfo data to collisionInfo
        collisionInfo = {
            'coordinate': Vec3(*collisionInfoData['coordinate']),
            'object': Object.fromDict(collisionInfoData['object']),
            'normal': Vec3(*collisionInfoData['normal']),
            'direction': collisionInfoData['direction'],
            'face': face,
        }
        
        return TraceTreeNode(ray, collisionInfo, leftNode, rightNode)

    def __repr__(self):
        return self.repr(self, 0)
            
    def repr(self, node, depth=0, index=-1):
        if node is None:
            return
        
        ## Create text representation of node with correct indent level
        indent = ' | ' * depth
        text = f"{indent}{(str(index)+': ') if index >= 0 else ''}⟬\u001b[36m{node.ray:.2f}\u001b[0m, \u001b[33m({node.collisionInfo['coordinate']:.2f}, {node.collisionInfo['object'].name, node.collisionInfo['object'].type}, {node.collisionInfo['normal']:.2f}, {node.collisionInfo['direction']}, {node.collisionInfo['face']})\u001b[0m⟭\n"
        
        ## Recursively get repr from next nodes
        if node.left is not None:
            text += self.repr(node.left, depth + 1, index=0)
        if node.right is not None:
            text += self.repr(node.right, depth + 1, index=1)
        
        return text


## RGB Color value
class RGB:
    def __init__(self, r=random.randint(0,255), g=random.randint(0,255), b=random.randint(0,255)) -> None:
        self.r = r
        self.g = g
        self.b = b
    
    def __add__(self, other):
        ## Add to other RGB
        if type(other) == RGB:
            return RGB(self.r + other.r, self.g + other.g, self.b + other.b).clamp()
        else:
            raise TypeError(f'Cannot add RGB to type {type(other)}')

    def __mul__(self, other):
        ## Multiply by number
        if type(other) in [int, float]:
            return RGB(self.r * other, self.g * other, self.b * other)
        ## Multiply by other RGB
        elif type(other) == RGB:
            return RGB(self.r * other.r, self.g * other.g, self.b * other.b)
        else:
            raise TypeError(f'Cannot multiply RGB with type {type(other)}')

    def __truediv__(self, other):
        ## Divide by number
        if type(other) in [int, float]:
            return RGB(self.r / other, self.g / other, self.b / other)
        ## Divide by other RGB
        elif type(other) == RGB:
            return RGB(self.r / other.r, self.g / other.g, self.b / other.b)
        else:
            raise TypeError(f'Cannot divide RGB by type {type(other)}')

    ## Get the mean average of multiple colors
    @staticmethod
    def mean(colors):
        num_colors = len(colors)
        
        ## Add all of the R G and B values
        total_r = sum([color.r for color in colors])
        total_g = sum([color.g for color in colors])
        total_b = sum([color.b for color in colors])
        
        ## Calculate the mean for each color R G and B
        mean_r = total_r // num_colors
        mean_g = total_g // num_colors
        mean_b = total_b // num_colors
        
        ## Return new RGB with the mean colors
        return RGB(mean_r, mean_g, mean_b).clamp()

    ## Make sure RGB value does not exceed (255, 255, 255)    
    def clamp(self):
        r = min(self.r, 255)
        g = min(self.g, 255)
        b = min(self.b, 255)

        return RGB(r, g, b)

    ## Convert to hexadecimal color code
    def toHex(self):
        return f'#{self.r:02x}{self.g:02x}{self.b:02x}'

    ## Convert to tuple in different color spaces
    def toTuple(self, space='rgb'):
        ## BGR mostly used for cv2
        if space == 'bgr':
            return (self.b, self.g, self.r)
        elif space == 'rgb':
            return (self.r, self.g, self.b)
        elif space == 'bgra':
            return (self.b, self.g, self.r, 255)
        elif space == 'rgba':
            return (self.r, self.g, self.b, 255)
        else:
            raise TypeError(f'Color space {space} not recognised')

    ## Use RGB as iterable
    def __iter__(self) -> iter:
        return iter([self.r, self.g, self.b])

    def __repr__(self) -> str:
        return f'【{self.r}, {self.g}, {self.b}】'

    def __format__(self, format_spec):
        if format_spec == 'h':
            ## Format as hex if h specifier
            return self.toHex()
        
        ## Format each part of the color based on the format specifier
        formatted_r = format(self.r, format_spec)
        formatted_g = format(self.g, format_spec)
        formatted_b = format(self.b, format_spec)
        
        return f'【{formatted_r}, {formatted_g}, {formatted_b}】'


## 3D Vector used for positional calculations
class Vec3:
    def __init__(self, x, y, z) -> None:
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, other):
        if type(other) in [float, int]:
            other = Vec3(other, other, other)
        if type(other) in [Vec3, Scale]:
            return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)
        else:
            raise TypeError(f'Cannot multiply Vec3 with type {type(other)}')
        
    def __truediv__(self, other):
        if type(other) in [float, int]:
            other = Vec3(other, other, other)
        if type(other) == Vec3:
            return Vec3(self.x / other.x, self.y / other.y, self.z / other.z)
        else:
            raise TypeError(f'Cannot divide Vec3 by type {type(other)}')

    ## Calculate the dot product of two vectors
    def dot(self, other):
        if type(other) == Vec3:
            ## Sum components
            return self.x * other.x + self.y * other.y + self.z * other.z
        else:
            raise TypeError(f'Cannot dot Vec3 against type {type(other)}')
    
    ## Calculate the component of a vector along the direction of another vector
    def component(self, other):
        if type(other) == Vec3:
            return self.dot(other.normalise())
        else:
            raise TypeError(f'Cannot get component of Vec3 along type {type(other)}')
    
    ## Project a vector onto another, like a shadow
    def project(self, other):
        if type(other) == Vec3:
            return self.component(other.normalise()) * other.normalise()
        else:
            raise TypeError(f'Cannot project Vec3 onto type {type(other)}')
    
    ## Calculate the magnitude of a vector with pythagoras
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    ## Calculate the cross product of self x other, in that order
    def cross(self, other):
        if type(other) == Vec3:
            return Vec3(
                self.y * other.z - self.z * other.y,
                self.z * other.x - self.x * other.z,
                self.x * other.y - self.y * other.x
            )
        else:
            raise TypeError(f'Cannot cross Vec3 against type {type(other)}')
    
    ## Apply the given transforms, returning a new vector
    def applyTransforms(self, position=None, scale=None, rotation=None):
        ## Set defaults
        position = position if position != None else Vec3(0, 0, 0)
        scale = scale if scale != None else Scale(0, 0, 0)
        rotation = rotation if rotation != None else Euler(0, 0, 0)
        
        ## Create rotation matrix based on euler
        rotationMatrix = rotation.toMatrix()
        
        ## Calculate transforms
        newVec = rotationMatrix * self
        newVec = newVec * scale
        newVec = newVec + position

        return newVec

    ## Normalise vector, making magnitude equal to 1
    def normalise(self):
        return self / self.magnitude()

    ## Check if vector is perpendicular to other vector
    def perpendicular(self, other):
        ## Account for floating point inacuracies by using small floats instead of zero
        return -0.000001 < self.dot(other) < 0.000001

    ## Check if vector is parallel to other vector
    def parallel(self, other):
        ## Account for floating point inacuracies by using small floats instead of zero
        return -0.000001 < self.cross(other).magnitude() < 0.000001

    ## Get the angle between two vectors
    def angle(self, other):
        dotProd = self.dot(other)
        magnitudesProd = self.magnitude() * other.magnitude()
        return math.acos(dotProd/magnitudesProd)

    ## Flip a vector over another vector
    def flip(self, direction):
        if type(direction) != Vec3:
            raise TypeError(f'Cannot flip Vec3 across type {type(direction)}')

        directionUnit = direction.normalise()
        projection = self.project(directionUnit)

        ## Subtract double the projection from the original vector
        flipped = self - (projection * 2)
        
        return flipped

    ## Use Vec3 as iterable
    def __iter__(self) -> iter:
        return iter([self.x, self.y, self.z])
    
    def __getitem__(self, index) -> int:
        return [self.x, self.y, self.z][index]
    
    def __setitem__(self, index, val) -> int:
        match index:
            case 0:
                self.x = val
            case 1:
                self.y = val
            case 2:
                self.z = val
    
    def __repr__(self) -> str:
        return f'⟦{self.x}, {self.y}, {self.z}⟧'
    
    def __format__(self, formatSpec: str) -> str:
        ## Format each component of the vector based on the format specifier
        formattedX = format(self.x, formatSpec)
        formattedY = format(self.y, formatSpec)
        formattedZ = format(self.z, formatSpec)
        
        return f'⟦{formattedX}, {formattedY}, {formattedZ}⟧'


## Scene structure used to collect objects together
class Scene:
    def __init__(self) -> None:
        self.objects = []
        self.camera = None

    ## Get an object given its id
    def getObject(self, id):
        for obj in self.objects:
            if obj.id == id:
                return obj
        
        ## No object found
        return None
    
    ## Convert scene to dictionary
    def toDict(self):
        ## Combine camera and objects data
        data = {
            'camera': self.camera.toDict(),
            'objects': [object.toDict() for object in self.objects]
        }
        
        return data
    
    ## Convert dictionary to scene
    @staticmethod
    def fromDict(data):
        ## Create an empty scene to add all of the scene data to
        scene = Scene()

        ## Create the camera
        cameraData = data.get('camera', {})
        camera = Camera.fromDict(cameraData)
        
        ## Set the camera to be in the given scene
        camera.scene = scene
        camera.id = cameraData.get('id', 0)
        
        ## Bind the camera
        scene.camera = camera
        
        scene.objects.append(camera)
        
        for obj in data.get('objects', []):
            ## If it is the currently bound camera, then ignore it (as it has already been set)
            if obj.get('boundCam', False):
                continue

            ## Create a new object
            object = Object.fromDict(obj)
            
            ## Add it to the scene
            scene.objects.append(object)

            ## Set the object to be in the given scene
            object.scene = scene
            object.id = obj.get('id', len(scene.objects))
        
        return scene

    ## Save the scene data to a JSON file
    def saveJSON(self, filepath):
        for object in self.objects:
            object.buildBVH()
        
        ## Get the dictionary representation of the scene
        data = self.toDict()
        
        ## Dump the data into JSON
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    ## Load a scene from a JSON file
    @staticmethod
    def fromJSON(filepath):
        ## Load the data from JSON
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        ## Convert the dictionary data into a scene
        scene = Scene.fromDict(data)
        
        return scene


## Shader structure providing information about an object's surface properties
class Shader:
    def __init__(self, name=None, debugColor=None, color=None, roughness=1.0, reflectivity=1.0, emissionStrength=0.0) -> None:
        self.name = name if name is not None else 'New shader'
        self.debugColor = debugColor if debugColor is not None else RGB(200, 200, 200)
        self.color = color if color is not None else RGB(200, 200, 200)
        self.roughness = roughness
        self.reflectivity = reflectivity
        self.emissionStrength = emissionStrength

    ## Convert shader to dictionary
    def toDict(self):
        return {
            'debugColor': [*self.debugColor],
            'color': [*self.color],
            'roughness': self.roughness,
            'reflectivity': self.reflectivity,
            'emissionStrength': self.emissionStrength,
        }


    ## Convert dictionary to shader
    @staticmethod
    def fromDict(data):
        return Shader(
            debugColor=RGB(*data.get('debugColor', [0, 0, 0])),
            color=RGB(*data.get('color', [0, 0, 0])),
            roughness=data.get('roughness', 1),
            reflectivity=data.get('reflectivity', 1),
            emissionStrength=data.get('emissionStrength', 0),
        )

    def __repr__(self) -> str:
        return f'debugColor: {self.debugColor}\ncolor: {self.color}\nroughness: {self.roughness}\nreflectivity: {self.reflectivity}\nemissionStrength: {self.emissionStrength}\n'


## Material structure providing information about an object's internal properties
class Material:
    def __init__(self, ior=1) -> None:
        self.ior = ior

    ## Convert material to dictionary
    def toDict(self):
        return {
            'ior': self.ior,
        }    

    ## Convert dictionary to material
    @staticmethod
    def fromDict(data):
        return Material(
            ior=data.get('ior', 1)
        )
        

## Mesh structure providing an object's faces and the vertices that they are made up of
class Mesh:
    def __init__(self, vertices, faces) -> None:
        self.vertices = vertices
        self.faces = faces

    ## Flip the normals of a mesh
    def flipNormals(self):
        ## Reverse order of all faces so that they go anticlockwise
        for i, face in enumerate(self.faces):
            self.faces[i] = face[::-1]
    
    ## Get the vertices associated with a face via indices
    def faceVertices(self, face):
        return [self.vertices[face[i]] for i in range(0, 3)]

    ## Get the centroid (center coordinate) of three vertices
    @staticmethod
    def centroid(v0, v1, v2):
        ## Calculate mean of each component to get centroid
        centroid = Vec3(
            (v0.x + v1.x + v2.x) / 3,
            (v0.y + v1.y + v2.y) / 3,
            (v0.z + v1.z + v2.z) / 3
        )

        return centroid

    ## Convert STL to mesh
    @staticmethod
    def fromSTL(filename):
        ## Use trimesh to load the mesh data
        mesh = trimesh.load_mesh(filename, enable_post_processing=True, solid=True)

        ## Vertices must be rearranged so that the y coordinate corresponds to height
        ## Faces should be converted to tuples
        vertices = [Vec3(float(vertex[0]), float(vertex[2]), float(vertex[1])) for vertex in mesh.vertices]
        faces = [tuple(face[::-1]) for face in mesh.faces]
        
        ## Create a new mesh from mesh data
        newMesh = Mesh(vertices, faces)
        
        return newMesh

    ## Convert mesh to dictionary
    def toDict(self):
        return {
            'vertices': [[*vec] for vec in self.vertices],
            'faces': [[int(vertex) for vertex in face] for face in self.faces],
        }

    ## Convert dictionary to mesh
    @staticmethod
    def fromDict(data):
        return Mesh(
            vertices=[Vec3(*vec) for vec in data.get('vertices', [])],
            faces=[tuple([int(vertex) for vertex in face]) for face in data.get('faces', [])],
        )

    def __repr__(self):
        ## Create empty return string
        returnString = ''
        
        ## Add all the vertices
        for vertex in self.vertices:
            returnString += str(vertex) + '\n'
        
        returnString += '\n'
        
        ## Add all the faces
        for face in self.faces:
            returnString += str(face) + '\n'
            
        return returnString


## Scale structure representing a 3-dimensional scale factor
class Scale:
    def __init__(self, x, y, z) -> None:
        self.x = x
        self.y = y
        self.z = z
    
    def __mul__(self, other):
        ## Multiply by number
        if type(other) in [float, int]:
            return Scale(self.x * other, self.y * other, self.z * other)
        ## Multiply by other scale
        elif type(other) == Scale:
            return Scale(self.x * other.x, self.y * other.y, self.z * other.z)
        else:
            raise TypeError(f'Cannot multiply Scale with type {type(other)}')
    
    def __truediv__(self, other):
        ## Divide by number
        if type(other) in [float, int]:
            return Scale(self.x / other, self.y / other, self.z / other)
        ## Divide by other scale
        elif type(other) == Scale:
            return Scale(self.x / other.x, self.y / other.y, self.z / other.z)
        else:
            raise TypeError(f'Cannot divide Scale by type {type(other)}')

    ## Use scale as iterable
    def __iter__(self) -> iter:
        return iter([self.x, self.y, self.z])
    
    def __getitem__(self, index) -> int:
        return [self.x, self.y, self.z][index]
    
    def __setitem__(self, index, val) -> int:
        match index:
            case 0:
                self.x = val
            case 1:
                self.y = val
            case 2:
                self.z = val
    
    def __repr__(self) -> str:
        return f'❰{self.x}, {self.y}, {self.z}❱'
    
    def __format__(self, formatSpec: str) -> str:
        ## Format each component of the vector based on the format specifier
        formattedX = format(self.x, formatSpec)
        formattedY = format(self.y, formatSpec)
        formattedZ = format(self.z, formatSpec)
        
        return f'❰{formattedX}, {formattedY}, {formattedZ}❱'


## Node of a Bounding Volume Hierarchy structure
class BVHNode:
    def __init__(self, box=(None, None), left=None, right=None, faces=None):
        self.box = box
        self.left = left
        self.right = right
        self.faces = faces

    ## Merge two BVH Nodes together
    def merge(self, other):
        ## Take the minimum of each component
        new_min = Vec3(
            min(self.box[0].x, other.box[0].x),
            min(self.box[0].y, other.box[0].y),
            min(self.box[0].z, other.box[0].z)
        )
        ## Take the maximum of each component
        new_max = Vec3(
            max(self.box[1].x, other.box[1].x),
            max(self.box[1].y, other.box[1].y),
            max(self.box[1].z, other.box[1].z)
        )

        return (new_min, new_max)

    ## Convert BVHNode to dictionary
    def toDict(self):
        ## Make sure faces are in int, not np.int64, if faces exist
        faces = [[int(vertex) for vertex in face] for face in self.faces] if self.faces != None else None
        
        ## Get dictionary representation of left and right nodes
        leftNode = self.left.toDict() if self.left else None
        rightNode = self.right.toDict() if self.right else None

        return {
            'box': [[*vec] if vec != None else None for vec in self.box],
            
            'left': leftNode,
            'right': rightNode,
            
            'faces': faces,
        }

    ## Convert dictionary to BVHNode
    @staticmethod
    def fromDict(data):
        try:
            ## Create nodes from left and right dictionaries
            loadedLeft = data.get('left', None)
            if loadedLeft != None:
                loadedLeft = BVHNode.fromDict(loadedLeft)
            loadedRight = data.get('right', None)
            if loadedRight != None:
                loadedRight = BVHNode.fromDict(loadedRight)
        except RecursionError:
            ## Load default BVH for left and right if recursion goes too deep
            loadedLeft = BVHNode()
            loadedRight = BVHNode()

        ## Combine data into BVHNode
        return BVHNode(
            box=[Vec3(*vec) if vec != None else None for vec in data.get('box', [None, None])],
            
            left=loadedLeft,
            right=loadedRight,
            
            faces=data.get('faces', None),
        )

    def __repr__(self) -> str:
        return f'[{self.box[0]}, {self.box[1]}], {self.left}, {self.right}, {self.faces[:4] if self.faces else "not a leaf"}\n'


class Object:
    def __init__(self, scene, name=None, mesh=None, position=None) -> None:
        self.__scene = scene
        self.id = max([-1] + [obj.id for obj in self.scene.objects]) + 1
        self.name = name if name is not None else 'Empty'
        self.position = position if position is not None else Vec3(0, 0, 0)
        self.rotation = Euler(0, 0, 0)
        self.scale = Scale(1, 1, 1)
        self.type = 'empty' if mesh is None else 'mesh'
        self.mesh = copy.deepcopy(mesh)
        self.shaders = []
        self.shaderIndices = []
        self.material = Material()
        self.bvh = BVHNode()
        
        self.scene.objects.append(self)
        self._defaultShaders()
    
    ## Load default shaders upon object creation
    def _defaultShaders(self):
        ## Create shader
        defaultShader = Shader()
        self.shaders = [defaultShader]

        ## Set shader indices depending on object type
        if self.mesh != None:
            self.shaderIndices = [0] * len(self.mesh.faces)
        elif self.type in ['light', 'sphere', 'camera', 'empty']:
            self.shaderIndices = [0]
        else:
            self.shaderIndices = []
    
    ## Build object's BVH based on mesh
    def buildBVH(self, faces=None, depth=0, maxTrianglesPerLeaf=8, maxDepth=100):
        ## Just get outer bounding box if object is not a mesh
        if self.type in ['sphere', 'light', 'empty', 'camera']:
            return BVHNode(box=self.boundingBox())
        
        ## Use object's faces if no list provided
        if faces == None: faces = copy.deepcopy(self.mesh.faces)

        box = self.boundingBox(faces)

        ## Get the size of the box on all axes
        diffs = []
        for axis in range(3):
            diff = abs(box[0][axis]-box[1][axis])
            diffs.append(diff)

        ## If the box is too small or contains too few faces, return the leaf node
        if len(faces) <= maxTrianglesPerLeaf or any(diff <= 0.0006 for diff in diffs) or depth >= maxDepth:
            return BVHNode(box=box, faces=faces)

        axis = depth % 3
        
        ## Sort the faces by their position on the axis
        faces.sort(key=lambda face: Mesh.centroid(*[self.mesh.vertices[index] for index in face])[axis])

        ## Split the faces into left and right halves
        mid = len(faces) // 2
        leftFaces = faces[:mid]
        rightFaces = faces[mid:]

        ## Get the node for each half recursively
        leftNode = self.buildBVH(leftFaces, depth+1, maxTrianglesPerLeaf, maxDepth)
        rightNode = self.buildBVH(rightFaces, depth+1, maxTrianglesPerLeaf, maxDepth)

        ## Combine the boxes into one bigger box
        boundingBox = leftNode.merge(rightNode)
        
        return BVHNode(box=boundingBox, left=leftNode, right=rightNode)
    
    ## Apply a shader to given faces
    def setShader(self, shader, faces=None):
        if faces == None:
            ## Apply shader to all faces
            if self.type in ['light', 'sphere', 'camera', 'empty']:
                ## If object is not a mesh, just change the first shader in the shaders list
                self.shaders[0] = shader
                self.shaderIndices = [0]
                return
            else:
                ## Get a list of all of the indices of the faces
                faces = list(range(len(self.mesh.faces)))

        if shader not in self.shaders:
            self.shaders.append(shader)

        shaderIndex = self.shaders.index(shader)
            
        for faceIndex in faces:
            self.shaderIndices[faceIndex] = shaderIndex
    
    ## Get the shader of a given face
    def getShader(self, face=None):
        shader = None
        if self.type == 'mesh':
            index = self.mesh.faces.index(face)
            shaderIndex = self.shaderIndices[index]
            shader = self.shaders[shaderIndex]
        elif self.type in ['light, camera', 'empty', 'sphere']:
            shader = self.shaders[0]
        return shader
    
    ## Apply the transforms to the mesh permanently
    def setTransforms(self):
        ## Setting transforms is irrelevant if not a mesh
        if self.type == 'mesh':
            ## Adjust the mesh's vertices list for the transformations
            newVertices = []
            for vertex in self.mesh.vertices:
                ## Get the vertex with transforms applied
                newVertex = vertex.applyTransforms(self.position, self.scale, self.rotation)
                newVertices.append(newVertex)
            self.mesh.vertices = newVertices

            ## Reset transformations
            self.position = Vec3(0, 0, 0)
            self.scale = Scale(1, 1, 1)
            self.rotation = Euler(0, 0, 0)

    ## Get the bounding box of a given set of faces
    def boundingBox(self, faces=None):
        if self.type == 'mesh':
            ## Use the object's faces if none provided
            if faces == None: faces = copy.deepcopy(self.mesh.faces)

            ## Get the transformed vertices from the faces
            vertices = [self.mesh.vertices[vertex] for face in faces for vertex in face]
            transformedVertices = [
                vertex.applyTransforms(self.position, self.scale, self.rotation) for vertex in vertices
            ]

            ## Get the min and max in all directions
            minX = min(vertex.x for vertex in transformedVertices) - 0.06
            minY = min(vertex.y for vertex in transformedVertices) - 0.06
            minZ = min(vertex.z for vertex in transformedVertices) - 0.06

            maxX = max(vertex.x for vertex in transformedVertices) + 0.06
            maxY = max(vertex.y for vertex in transformedVertices) + 0.06
            maxZ = max(vertex.z for vertex in transformedVertices) + 0.06

            return Vec3(minX, minY, minZ), Vec3(maxX, maxY, maxZ)
        elif self.type in ['sphere', 'light']:
            minVec = self.position - Vec3(self.scale.x/2, self.scale.x/2, self.scale.x/2) - Vec3(0.06, 0.06, 0.06)
            maxVec = self.position + Vec3(self.scale.x/2, self.scale.x/2, self.scale.x/2) + Vec3(0.06, 0.06, 0.06)
            
            return minVec, maxVec
        elif self.type in ['empty', 'camera']:
            return self.position, self.position
        else:
            return None
    
    ## Convert object to dictionary
    def toDict(self):
        ## Standard data for all types of object
        data = {
            'id': self.id,

            'name': self.name,
            'type': self.type,
            
            'position': [*self.position],
            'rotation': [*self.rotation],
            'scale': [*self.scale],
            
            'shaders': [shader.toDict() for shader in self.shaders],
            'shaderIndices': self.shaderIndices,
            
            'material': self.material.toDict(),
            
            'bvh': self.bvh.toDict(),
        }

        ## Specific sets of data for different types
        if self.type == 'mesh':
            return {
                **data,
                'mesh': self.mesh.toDict(),
            }
        if self.type == 'camera':
            cameraData = {
                'length': self.length,
                'boundCam': self == self.scene.camera,
            }
            
            return {
                **data,
                **cameraData,
            }

        if self.type == 'light':
            return {
                **data,
                'strength': self.strength,
            }
        
        return data
    
    ## Convert dictionary to object
    @staticmethod
    def fromDict(data): # NOTE ## Objects are loaded into a new empty scene if loaded individually. To load into a given scene, either load the scene itself, or load the object and then reset its scene property to the correct scene.
        type = data.get('type', 'empty')

        ## Create object depending on type
        if type == 'mesh':
            obj = Object(scene=Scene())
            obj.mesh = Mesh.fromDict(data.get('mesh', {'vertices': [], 'faces': []}))
        elif type == 'camera':
            obj = Camera(scene=Scene())
            obj.length = data.get('length', 1)
        elif type == 'light':
            obj = Light(scene=Scene())
            obj.strength = data.get('strength', 1)
        elif type == 'sphere':
            obj = Sphere(scene=Scene())
        else:
            obj = Object(scene=Scene())

        ## Set object data
        obj.name = data.get('name', 'Empty Object')
        obj.type = type
        
        obj.position = Vec3(*data.get('position', [0, 0, 0]))
        obj.rotation = Euler(*data.get('rotation', [0, 0, 0]))
        obj.scale = Scale(*data.get('scale', [0, 0, 0]))
        
        obj.shaders = [Shader.fromDict(shaderDict) for shaderDict in data.get('shaders', [])]
        obj.shaderIndices = data.get('shaderIndices', [])
        
        obj.material = Material.fromDict(data.get('material', {}))
        
        obj.bvh = BVHNode.fromDict(data.get('bvh', {}))
        
        return obj
    
    ## Load shader JSON data as a shader
    def loadShadersFromJSON(self, filepath):
        with open(filepath, 'r') as file:
            data = json.load(file)
        
        ## Get shaders and indices list
        shadersJSON= data.get("shaders", {})
        facesJSON = data.get("faces", {})
        
        ## Clip list if too long
        if len(facesJSON) > len(self.mesh.faces):
            facesJSON = facesJSON[:len(self.mesh.faces)]

        self.shaders = [shader for shader in shadersJSON.values()]
        self.shaderIndices = [shaderIndex for shaderIndex in facesJSON.values()]
    
    ## Scene property to properly assign id when reassigning scene
    @property
    def scene(self):
        return self.__scene
    @scene.setter
    def scene(self, scene):
        self.__scene = scene
        self.id = len(self.__scene.objects)


## Preset cube (just an empty object with a standard mesh)
class Cube(Object):
    def __init__(self, scene, name='Cube', position=None) -> None:
        super(Cube, self).__init__(scene, name=name, position=position)
        self.type = 'mesh'
        self.mesh = Mesh([ ## Vertices
                Vec3(1, 1, 1),
                Vec3(-1, 1, 1),
                Vec3(-1, -1, 1),
                Vec3(1, -1, 1),
                Vec3(1, 1, -1),
                Vec3(-1, 1, -1),
                Vec3(-1, -1, -1),
                Vec3(1, -1, -1)
            ],
            [ ## Faces
                (0, 1, 2),
                (2, 3, 0),
                (0, 3, 7),
                (7, 4, 0),
                (6, 2, 1),
                (1, 5, 6),
                (5, 1, 0),
                (0, 4, 5),
                (3, 2, 6),
                (6, 7, 3),
                (6, 5, 4),
                (4, 7, 6),
            ])
        self.bvh = self.buildBVH()
        super(Cube, self)._defaultShaders()


## Preset sphere object (represented by mathematical sphere, not mesh)
class Sphere(Object):
    def __init__(self, scene, name='Sphere', position=None) -> None:
        super(Sphere, self).__init__(scene, name=name, position=position)
        self.type = 'sphere'
        self.bvh = self.buildBVH()
        super(Sphere, self)._defaultShaders()


## Preset camera object with length information
class Camera(Object):
    def __init__(self, scene, name='Camera', length=20, position=None) -> None:
        super(Camera, self).__init__(scene, name=name, position=position)
        self.__length = length/20
        self.type = 'camera'
        if self.scene.camera == None:
            self.scene.camera = self
        self.bvh = self.buildBVH()
        super(Camera, self)._defaultShaders()
    
    @property
    def length(self):
        return self.__length*20
    @length.setter
    def length(self, length):
        self.__length = length/20
            

## Preset light object with strength information
class Light(Object):
    def __init__(self, scene, name='Light', position=None, strength=1) -> None:
        super(Light, self).__init__(scene, name=name, position=position)
        self.type = 'light'
        self.strength = strength
        self.bvh = self.buildBVH()
        super(Light, self)._defaultShaders()