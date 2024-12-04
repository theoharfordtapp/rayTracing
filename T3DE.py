# AQA A-Level Computer Science NEA 2024
#
# Graphics Engine
# Theo HT
#
# (3D Engine)
#
# NOTE # Comments with `MARK` in them are purely IDE-related. They have no relevance to the code.

# MARK: IMPORTS
import math, random
import trimesh
import json
import copy


# MARK: MATRIX
## 9x9 matrix used to calculate rotations
class Matrix9:
    def __init__(self, elements):
        self.elements = elements
    
    # MARK: > ARITHMETIC
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


# MARK: EULER
## Euler angle used to save rotation information
class Euler:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    # MARK: > ADMIN
    ## Convert to iterable
    def __iter__(self) -> iter:
        return iter([self.x, self.y, self.z])

    ## MARK: > REPR
    def __repr__(self) -> str:
        return f'◿ {self.x}, {self.y}, {self.z} ◺'
    
    def __format__(self, formatSpec: str) -> str:
        ## Format each component of the vector based on the format specifier
        formattedX = format(self.x, formatSpec)
        formattedY = format(self.y, formatSpec)
        formattedZ = format(self.z, formatSpec)
        
        return f'◿ {formattedX}, {formattedY}, {formattedZ} ◺'

    # MARK: > ARITHMETIC
    def __mul__(self, other):
        ## Multiply by number
        if type(other) in [float, int]:
            return Euler(self.x * other, self.y * other, self.z * other)
        else:
            raise TypeError(f'Cannot multiply Euler with type {type(other)}')
    
    def __add__(self, other):
        ## Add to other Euler
        if type(other) == Euler:
            return Euler(self.x + other.x, self.y + other.y, self.z + other.z)
        else:
            raise TypeError(f'Cannot add type {type(other)} to Euler')
    
    # MARK: > TO MATRIX
    ## Convert to matrix
    def toMatrix(self):
        ## Get trig values
        cos_x, sin_x = math.cos(self.x), math.sin(self.x)
        cos_y, sin_y = math.cos(self.y), math.sin(self.y)
        cos_z, sin_z = math.cos(self.z), math.sin(self.z)

        ## Create matrices
        Rx = Matrix9([
            1, 0, 0,
            0, cos_x, -sin_x,
            0, sin_x, cos_x
        ])
        Ry = Matrix9([
            cos_y, 0, sin_y,
            0, 1, 0,
            -sin_y, 0, cos_y
        ])
        Rz = Matrix9([
            cos_z, -sin_z, 0,
            sin_z, cos_z, 0,
            0, 0, 1
        ])

        ## Combine matrices
        return Rz * Ry * Rx


# MARK: RAY
## Ray used for tracing paths
class Ray:
    def __init__(self, start, direction, ior, object) -> None:
        self.start = start
        self.direction = direction.normalise()
        self.ior = ior
        self.object = object
    
    # MARK: > REPR
    def __repr__(self) -> str:
        return f'<{self.start}, {self.direction}>'
    
    def __format__(self, formatSpec) -> str:
        formattedStart = format(self.start, formatSpec)
        formattedDirection = format(self.direction, formatSpec)
        return f'<{formattedStart}, {formattedDirection}>'
    
    # MARK: > TO VEC3
    ## Convert ray to normalised vector
    def toVec3(self):
        return (self.pointOnRay(1) - self.start).normalise()
    
    # MARK: > TO DICT
    ## Convert ray to a dictionary that can be stored for later use
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
    
    # MARK: > FROM DICT
    ## Convert dictionary back to a ray
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
    
    # MARK: > GET T
    ## Get distance along ray where a collision with a plane occurs
    def getT(self, plane):
        ## Get ray and plane information
        startPoint = self.start
        rayDirection = self.direction
        normal = plane.normal
        planePos = plane.position
        
        ## Simultaneous equation combining two equations: 1) check if collision is on plane, 2) get the distance along a ray for a certain collision
        ## Calculates what point along the ray would be on the plane
        t = (normal.dot(startPoint) - normal.dot(planePos))/((normal * -1).dot(rayDirection))
        return t

    ## Get distance along ray where a collision with a sphere occurs
    def getTSphere(self, sphere):
        ## Get ray and sphere information
        startPoint = self.start
        spherePos = sphere.position
        startPointAbsolute = startPoint - spherePos
        rayDirection = self.direction
        sphereRad = sphere.radius
        
        ## Discriminant of the equation which solves for the point where a line and a sphere touch
        ## If discriminant is positive, it touches twice (passes through the sphere)
        ## If discriminant is 0, it touches once (just touches the edge of the sphere)
        discriminant = (startPointAbsolute.dot(rayDirection)**2) - (startPointAbsolute.dot(startPointAbsolute)-(sphereRad**2))
        
        ## If discriminant is negative, it never touches the sphere
        if discriminant < 0: return None

        ## Get the distances along the ray (t) if at least one exists. t1 represents the first collision, t2 the second
        t1 = (startPointAbsolute.dot(rayDirection) * -1) - math.sqrt(discriminant)
        t2 = (startPointAbsolute.dot(rayDirection) * -1) + math.sqrt(discriminant)

        return (t1, t2)

    # MARK: CHECK AABB
    ## Check if a ray ever enters a bounding box
    def entersAABB(self, box):
        tAxes = [0, 0, 0]
        
        ## Iterate through all 3 axes
        for axis in range(3):
            ## Get the component of the origin and direction vectors along this axis
            originComponent = self.start[axis]
            directionComponent = self.direction[axis]
            
            ## Ray is perpendicular to this axis (i.e. parallel to the faces of the box that lie perpendicular to this axis)
            if directionComponent == 0:
                ## Ray starts outside the box
                if originComponent < box[0][axis] or originComponent > box[1][axis]:
                    ## If the ray is perpendicular to this axis and starts without the box on this axis, then it can never intersect with the box at all
                    return False
                
                ## If the ray is perpendicular but starts within the box on this axis, it will never intersect with the box on this axis
                ## But it might intersect on the other axes, so we say it intersects with this axis at an infinite distance in both directions
                tAxes[axis] = (float('-inf'), float('inf'))
                continue
            
            ## Get the component of the box's min and max coordinates on this axis
            boxMinComponent = box[0][axis]
            boxMaxComponent = box[1][axis]
            
            ## Get the distance along the ray at which point the ray intersects with the box's min and max coordinates on this axis
            tMin = (boxMinComponent - originComponent)/directionComponent
            tMax = (boxMaxComponent - originComponent)/directionComponent
            
            ## Swap if the ray is going the other direction
            if tMin > tMax:
                tMin, tMax = tMax, tMin
            
            tAxes[axis] = (tMin, tMax)
        
        ## Calculate when the ray enters and exits the box
        tEnter = max(tAxis[0] for tAxis in tAxes)
        tExit = min(tAxis[1] for tAxis in tAxes)
        
        ## Return if the ray enters the box before it exists and if it exits in a positive direction
        return tEnter <= tExit and tExit > 0

    # MARK: > POINT
    ## Convert t distance into absolute vector
    def pointOnRay(self, t):
        return self.start + (self.direction*t)


# MARK: MATH OBJS
## Mathematically-represented plane object, instead of mesh
class MathematicalPlane:
    def __init__(self, position, normal) -> None:
        self.position = position
        self.normal = normal


## Mathematically-represented sphere object, instead of mesh approximation
class MathematicalSphere:
    def __init__(self, position, radius) -> None:
        self.position = position
        self.radius = radius


# MARK: TRACE
## Node of a ray trace tree, used to store information about a ray trace
class TraceTreeNode:
    def __init__(self, ray, collisionInfo, left=None, right=None) -> None:
        self.ray = ray
        self.collisionInfo = collisionInfo
        self.left = left if left is not None else None
        self.right = right if right is not None else None

    # MARK: > REPR
    ## (Recursive)
    def __repr__(self):
        return self.repr(self, 0)
            
    def repr(self, node, depth=0, index=-1):
        ## Base case
        if node is None:
            return
        
        ## Create text representation of node with correct indent level
        indent = ' | ' * depth
        text = f"{indent}{(str(index)+': ') if index >= 0 else ''}⟬\u001b[36m{node.ray:.2f}\u001b[0m, \u001b[33m({node.collisionInfo['coordinate']:.2f}, {node.collisionInfo['object'].name, node.collisionInfo['object'].type}, {node.collisionInfo['normal']:.2f}, {node.collisionInfo['direction']}, {node.collisionInfo['face']})\u001b[0m⟭\n"
        
        ## Recursion
        if node.left is not None:
            text += self.repr(node.left, depth + 1, index=0)
        if node.right is not None:
            text += self.repr(node.right, depth + 1, index=1)
        
        return text
    
    # MARK: > TO DICT
    ## Convert trace tree node to a dictionary that can be stored for later use
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
    
    # MARK: > FROM DICT
    ## Convert trace tree node data back into a trace tree node
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


# MARK: RGB
## RGB Color value
class RGB:
    def __init__(self, r=random.randint(0,255), g=random.randint(0,255), b=random.randint(0,255)) -> None:
        self.r = r
        self.g = g
        self.b = b
    
    # MARK: > ADMIN
    ## Convert to iterable
    def __iter__(self) -> iter:
        return iter([self.r, self.g, self.b])
    
    # MARK: > REPR
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

    def pprint(self):
        print('\tR: ' + str(self.r))
        print('\tG: ' + str(self.g))
        print('\tB: ' + str(self.b))
        print('\tHex: ' + self.toHex())
    
    # MARK: > ARITHMETIC
    def __mul__(self, other):
        if type(other) in [int, float]:
            ## Multiply each component by other
            return RGB(self.r * other, self.g * other, self.b * other)
        elif type(other) == RGB:
            ## Multiply component-wise
            return RGB(self.r * other.r, self.g * other.g, self.b * other.b)
        else:
            raise TypeError(f'Cannot multiply RGB with type {type(other)}')
    
    def __add__(self, other):
        if type(other) == RGB:
            r = min(self.r + other.r, 255)
            g = min(self.g + other.g, 255)
            b = min(self.b + other.b, 255)
            return RGB(r, g, b)
        else:
            raise TypeError(f'Cannot add RGB to type {type(other)}')

    # MARK: > TO HEX
    ## Convert to hexadecimal color code
    def toHex(self):
        return f'#{self.r:02x}{self.g:02x}{self.b:02x}'
    
    # MARK: > TO TUPLE
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
    
    # MARK: > MEAN
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

    # MARK: > CLAMP
    ## Make sure RGB value does not exceed (255, 255, 255)    
    def clamp(self):
        r = min(self.r, 255)
        g = min(self.g, 255)
        b = min(self.b, 255)
        
        return RGB(r, g, b)


## MARK: VEC3
# 3D Vector used for positional calculations
class Vec3:
    def __init__(self, x, y, z) -> None:
        self.x = x
        self.y = y
        self.z = z
    
    # MARK: ADMIN
    def __iter__(self) -> iter:
        return iter([self.x, self.y, self.z])
    
    def __getitem__(self, index) -> int:
        return [self.x, self.y, self.z][index]
    
    ## MARK: > REPR
    def __repr__(self) -> str:
        return f'⟦{self.x}, {self.y}, {self.z}⟧'
    
    def __format__(self, formatSpec: str) -> str:
        ## Format each component of the vector based on the format specifier
        formattedX = format(self.x, formatSpec)
        formattedY = format(self.y, formatSpec)
        formattedZ = format(self.z, formatSpec)
        
        return f'⟦{formattedX}, {formattedY}, {formattedZ}⟧'

    # MARK: > ARITHMETIC
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

    # MARK: > DOT
    ## Calculate the dot product of two vectors
    def dot(self, other):
        if type(other) == Vec3:
            ## Return sum of product of each component
            return self.x * other.x + self.y * other.y + self.z * other.z
        else:
            raise TypeError(f'Cannot dot Vec3 against type {type(other)}')
    
    # MARK: > COMPONENT
    ## Calculate the component of a vector along the direction of another vector
    def component(self, other):
        if type(other) == Vec3:
            return self.dot(other.normalise())
        else:
            raise TypeError(f'Cannot get component of Vec3 along type {type(other)}')
    
    # MARK: > PROJECT
    ## Project a vector onto another, like a shadow
    def project(self, other):
        if type(other) == Vec3:
            return other.normalise() * self.component(other.normalise())
        else:
            raise TypeError(f'Cannot project Vec3 onto type {type(other)}')
    
    # MARK: > MAGNITUDE
    ## Calculate the magnitude of a vector with pythagoras
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    # MARK: > CROSS
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
    
    # MARK: > TRANSFORM
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
    
    ## Set the given transforms, applying them permanently to self
    def setTransforms(self, position=None, scale=None, rotation=None):
        ## Set defaults
        position = position if position != None else Vec3(0, 0, 0)
        scale = scale if scale != None else Scale(0, 0, 0)
        rotation = rotation if rotation != None else Euler(0, 0, 0)
        
        ## Get the vector with transforms applied
        newVec = self.applyTransforms(position, scale, rotation)

        ## Set the new coordinates
        self.x = newVec.x
        self.y = newVec.y
        self.z = newVec.z

    # MARK: > NORMALISE
    ## Normalise vector, making magnitude equal to 1
    def normalise(self):
        return self / self.magnitude()

    # MARK: > PERPENDICULAR
    ## Check if vector is perpendicular to other vector
    def perpendicular(self, other):
        ## Account for floating point inacuracies by using small floats instead of zero
        return -0.000001 < self.dot(other) < 0.000001

    # MARK: > PARALLEL
    ## Check if vector is parallel to other vector
    def parallel(self, other):
        ## The magnitude of the cross product is smaller the closer the two vectors are to parallel
        ## Account for floating point inacuracies by using small floats instead of zero
        return -0.000001 < self.cross(other).magnitude() < 0.000001
    
    # MARK: > ANGLE
    ## Get the angle between two vectors
    def angle(self, other):
        dotProd = self.dot(other)
        magnitudesProd = self.magnitude() * other.magnitude()
        return math.acos(dotProd/magnitudesProd)

    # MARK: > FLIP
    ## Flip a vector over another vector
    def flipComponent(self, direction):
        if type(direction) != Vec3:
            raise TypeError(f'Cannot flip Vec3 across type {type(direction)}')

        ## Get direction unit vector
        directionUnit = direction.normalise()

        ## Get the projection (shadow) of the ray on the direction vector
        projection = self.project(directionUnit)

        ## Subtract two times the projection from the original vector to get the flipped vector
        flipped = self - (projection * 2)
        
        return flipped


# MARK: SCENE
## Scene structure used to collect the objects information
class Scene:
    def __init__(self) -> None:
        self.objects = []
        self.camera = None
    
    # MARK: > GET OBJ
    ## Get an object given its id
    def getObject(self, id):
        for obj in self.objects:
            if obj.id == id:
                return obj
        
        ## No object found
        return None
    
    # MARK: > TO DICT
    ## Convert scene data to dict to save
    def toDict(self):
        ## Combine camera and objects data
        data = {
            'camera': self.camera.toDict(),
            'objects': [object.toDict() for object in self.objects]
        }
        
        return data
    
    # MARK: > FROM DICT
    ## Load scene data from dict
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

    # MARK > SAVE JSON
    ## Save the scene data to a JSON file
    def saveJSON(self, filepath):
        ## Get the dictionary representation of the scene
        data = self.toDict()
        
        ## Dump the data into JSON
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    # MARK > FROM JSON
    ## Create a scene from a JSON file
    @staticmethod
    def fromJSON(filepath):
        ## Load the data from JSON
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        ## Convert the dictionary data into a scene
        scene = Scene.fromDict(data)
        
        return scene

# MARK: SHADER
## Shader structure providing information about an object's surface properties
class Shader:
    def __init__(self, debugColor=None, color=None, roughness=1, reflectivity=1, emissionStrength=0) -> None:
        self.debugColor = debugColor if debugColor is not None else RGB(200, 200, 200)
        self.color = color if color is not None else RGB(200, 200, 200)
        self.roughness = roughness
        self.reflectivity = reflectivity
        self.emissionStrength = emissionStrength
    
    # MARK: > TO DICT
    ## Convert shader data to dictionary to save
    def toDict(self):
        return {
            'debugColor': [*self.debugColor],
            'color': [*self.color],
            'roughness': self.roughness,
            'reflectivity': self.reflectivity,
            'emissionStrength': self.emissionStrength,
        }

    # MARK: > FROM DICT
    ## Get dictionary data and convert to shader
    @staticmethod
    def fromDict(data):
        return Shader(
            debugColor=RGB(*data.get('debugColor', [0, 0, 0])),
            color=RGB(*data.get('color', [0, 0, 0])),
            roughness=data.get('roughness', 1),
            reflectivity=data.get('reflectivity', 1),
            emissionStrength=data.get('emissionStrength', 0),
        )

    # MARK: > REPR
    def __repr__(self) -> str:
        return f'debugColor: {self.debugColor}\ncolor: {self.color}\nroughness: {self.roughness}\nreflectivity: {self.reflectivity}\nemissionStrength: {self.emissionStrength}\n'

# MARK: MATERIAL
## Matierial structure providing information about an object's internal properties
class Material:
    def __init__(self, ior=1) -> None:
        self.ior = ior
    
    # MARK: > TO DICT
    ## Convert material data to dictionary to save
    def toDict(self):
        return {
            'ior': self.ior,
        }    

    # MARK: > FROM DICT
    ## Get dictionary data and convert to material
    @staticmethod
    def fromDict(data):
        return Material(
            ior=data.get('ior', 1)
        )
        

# MARK: MESH
## Mesh structure providing an object's faces and the vertices that they are made up of
class Mesh:
    def __init__(self, vertices, faces) -> None:
        self.vertices = vertices
        self.faces = faces

    # MARK: > FROM STL
    ## Convert STL to a mesh
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

    # MARK: > CENTROID
    ## Get the centroid (center coordinate) of three vertices
    @staticmethod
    def centroid(vertices, face):
        ## Get the vertex coordinates from the face made up of vertex indices
        v0, v1, v2 = [vertices[vertex] for vertex in face]

        ## Calculate mean of each component to get centroid
        centroid = Vec3(
            (v0.x + v1.x + v2.x) / 3,
            (v0.y + v1.y + v2.y) / 3,
            (v0.z + v1.z + v2.z) / 3
        )

        return centroid

    # MARK: > REPR
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
    
    # MARK: > FLIP NORMALS
    ## Flip the normals of a mesh
    def flipNormals(self):
        ## Reverse order of all faces so that they go anticlockwise
        for i, face in enumerate(self.faces):
            self.faces[i] = face[::-1]
    
    # MARK: > TO DICT
    ## Convert mesh to dictionary data to store for later use
    def toDict(self):
        return {
            'vertices': [[*vec] for vec in self.vertices],
            'faces': [[int(vertex) for vertex in face] for face in self.faces],
        }

    # MARK: > FROM DICT
    ## Convert dictionary data back to a mesh
    @staticmethod
    def fromDict(data):
        return Mesh(
            vertices=[Vec3(*vec) for vec in data.get('vertices', [])],
            faces=[tuple([int(vertex) for vertex in face]) for face in data.get('faces', [])],
        )


# MARK: SCALE
## Scale structure representing a 3-dimensional scale factor
class Scale:
    def __init__(self, x, y, z) -> None:
        self.x = x
        self.y = y
        self.z = z
    
    # MARK: > ADMIN
    def __iter__(self) -> iter:
        return iter([self.x, self.y, self.z])
    
    # MARK: > REPR
    def __repr__(self) -> str:
        return f'❰{self.x}, {self.y}, {self.z}❱'
    
    def __format__(self, formatSpec: str) -> str:
        ## Format each component of the vector based on the format specifier
        formattedX = format(self.x, formatSpec)
        formattedY = format(self.y, formatSpec)
        formattedZ = format(self.z, formatSpec)
        
        return f'❰{formattedX}, {formattedY}, {formattedZ}❱'
    
    # MARK: > ARITHMETIC
    def __truediv__(self, other):
        if type(other) in [float, int]:
            return Scale(self.x/other, self.y/other, self.z/other)
        else:
            raise TypeError(f'Cannot divide Scale by type {type(other)}')
    
    def __mul__(self, other):
        if type(other) in [float, int]:
            return Scale(self.x * other, self.y * other, self.z * other)
        elif type(other) == Scale:
            return Scale(self.x * other.x, self.y * other.y, self.z * other.z)
        else:
            raise TypeError(f'Cannot multiply Scale with type {type(other)}')


# MARK: BVHNODE
## Node of a Bounding Volume Hierarchy structure
class BVHNode:
    def __init__(self, box=(None, None), left=None, right=None, faces=None):
        self.box = box
        self.left = left
        self.right = right
        self.faces = faces

    # MARK: > MERGE
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
    
    # MARK: > REPR
    def __repr__(self) -> str:
        return f'[{self.box[0]}, {self.box[1]}], {self.left}, {self.right}, {self.faces[:4] if self.faces else "not a leaf"}\n'

    # MARK: > TO DICT
    ## Convert BVHNode to dictionary data to save and store for later use
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
    
    # MARK: > FROM DICT
    ## Convert dictionary data to BVHNode
    @staticmethod
    def fromDict(data):
        try:
            loadedLeft = data.get('left', {})
            if loadedLeft != None:
                loadedLeft = BVHNode.fromDict(loadedLeft)
            loadedRight = data.get('right', {})
            if loadedRight != None:
                loadedRight = BVHNode.fromDict(loadedRight)
        except RecursionError:
            loadedLeft = BVHNode()
            loadedRight = BVHNode()

        return BVHNode(
            box=[Vec3(*vec) if vec != None else None for vec in data.get('box', [None, None])],
            
            left=loadedLeft,
            right=loadedRight,
            
            faces=data.get('faces', None),
        )


class Object:
    def __init__(self, scene, name=None, mesh=None, position=None) -> None:
        self.__scene = scene
        self.id = len(self.scene.objects)
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
    
    @property
    def scene(self):
        return self.__scene
    
    @scene.setter
    def scene(self, scene):
        self.__scene = scene
        self.id = len(self.__scene.objects)
    
    def _defaultShaders(self):
        defaultShader = Shader()
        self.shaders = [defaultShader]
        if self.mesh != None:
            self.shaderIndices = [0] * len(self.mesh.faces)
        elif self.type in ['light', 'sphere', 'camera', 'empty']:
            self.shaderIndices = [0]
        else:
            self.shaderIndices = []
    
    def setShader(self, shader, faces=None):
        if faces == None:
            if self.type in ['light', 'sphere', 'camera', 'empty']:
                self.shaders[0] = shader
                self.shaderIndices = [0]
                return
            else:
                faces = list(range(len(self.mesh.faces)))

        if shader not in self.shaders:
            self.shaders.append(shader)
        shaderIndex = self.shaders.index(shader)
            
        for faceIndex in faces:
            self.shaderIndices[faceIndex] = shaderIndex
    
    def loadShadersFromJSON(self, filepath):
        with open(filepath, 'r') as file:
            data = json.load(file)

        shadersJSON= data.get("shaders", {})
        facesJSON = data.get("faces", {})
        if len(facesJSON) > len(self.mesh.faces):
            facesJSON = facesJSON[:len(self.mesh.faces)]
        
        self.shaders = [shader for shader in shadersJSON.values()]
        self.shaderIndices = [shaderIndex for shaderIndex in facesJSON.values()]
    
    def toDict(self):
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
    
    @staticmethod
    def fromDict(data): # NOTE ## Objects are loaded into a new empty scene if loaded individually. To load into a given scene, either load the scene itself, or load the object and then reset its scene property to the correct scene.
        type = data.get('type', 'empty')

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
    
    def setTransforms(self):
        if self.type == 'mesh':
            for vertex in self.mesh.vertices:
                vertex.setTransforms(self.position, self.scale, self.rotation)

            self.position = Vec3(0, 0, 0)
            self.scale = Scale(1, 1, 1)
            self.rotation = Euler(0, 0, 0)
            

    def boundingBox(self, faces=None):
        if self.type == 'mesh':
            if faces == None: faces = copy.deepcopy(self.mesh.faces)
            vertices = [self.mesh.vertices[vertex] for face in faces for vertex in face]
            transformedVertices = [
                vertex.applyTransforms(self.position, self.scale, self.rotation) for vertex in vertices
            ]

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
        elif self.type == 'empty':
            return self.position, self.position
        else:
            return None

    
    def buildBVH(self, faces=None, depth=0, maxTrianglesPerLeaf=8, maxDepth=100):
        if self.type in ['sphere', 'light', 'empty', 'camera']:
            return BVHNode(box=self.boundingBox())
        
        if faces == None: faces = copy.deepcopy(self.mesh.faces)

        box = self.boundingBox(faces)

        diffs = [abs(box[0][axis]-box[1][axis]) for axis in range(3)]

        if len(faces) <= maxTrianglesPerLeaf or any(diff <= 0.0006 for diff in diffs) or depth >= maxDepth:
            return BVHNode(box=box, faces=faces)

        axis = depth % 3
        
        faces.sort(key=lambda face: Mesh.centroid(self.mesh.vertices, face)[axis])

        mid = len(faces) // 2
        leftFaces = faces[:mid]
        rightFaces = faces[mid:]

        leftNode = self.buildBVH(leftFaces, depth+1, maxTrianglesPerLeaf, maxDepth)
        rightNode = self.buildBVH(rightFaces, depth+1, maxTrianglesPerLeaf, maxDepth)

        boundingBox = leftNode.merge(rightNode)
        
        return BVHNode(box=boundingBox, left=leftNode, right=rightNode)


class Cube(Object):
    def __init__(self, scene, name='Cube', position=None) -> None:
        super(Cube, self).__init__(scene, name=name, position=position)
        self.type = 'mesh'
        self.mesh = Mesh([ # vertices
                Vec3(1, 1, 1),
                Vec3(-1, 1, 1),
                Vec3(-1, -1, 1),
                Vec3(1, -1, 1),
                Vec3(1, 1, -1),
                Vec3(-1, 1, -1),
                Vec3(-1, -1, -1),
                Vec3(1, -1, -1)
            ],
            [ # faces
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
        super(Cube, self)._defaultShaders()

class Sphere(Object):
    def __init__(self, scene, name='Sphere', position=None) -> None:
        super(Sphere, self).__init__(scene, name=name, position=position)
        self.type = 'sphere'
        super(Sphere, self)._defaultShaders()

class Camera(Object):
    def __init__(self, scene, name='Camera', length=20, position=None) -> None:
        super(Camera, self).__init__(scene, name=name, position=position)
        self.__length = length/20
        self.type = 'camera'
        if self.scene.camera == None:
            self.scene.camera = self
        super(Camera, self)._defaultShaders()
    
    @property
    def length(self):
        return self.__length*20
    
    @length.setter
    def length(self, length):
        self.__length = length/20
            
class Light(Object):
    def __init__(self, scene, name='Light', position=None, strength=1) -> None:
        super(Light, self).__init__(scene, name=name, position=position)
        self.type = 'light'
        self.bvh = self.buildBVH()
        self.strength = strength
        super(Light, self)._defaultShaders()