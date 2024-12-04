from T3DE import Scene, Object, Mesh, Scale, Shader, Light, Camera, Vec3, RGB, Cube, Sphere, Material
from T3DR import Photon, Debug, Texel, Post
import math
import json
import cv2

scene: Scene = Scene()

boxMesh: Mesh = Mesh.fromSTL('cornellbox.stl')
monkeyMesh: Mesh = Mesh.fromSTL('suzanne.stl')

box: Object = Object(scene=scene, name='Box', mesh=boxMesh)
# box.rotation.x = math.radians(-90)
box.rotation.y = math.radians(90)
box.setTransforms()
boxShader = Shader(
    reflectivity = 1,
    roughness = 1,
    color = RGB(200, 200, 200),
    debugColor = RGB(100, 100, 100)
)
boxEmissionShader = Shader(
    color = RGB(255, 255, 120),
    debugColor = RGB(120, 120, 20),
    emissionStrength = 1
)
redWallShader = Shader(
    reflectivity = 1,
    roughness = 0.68,
    color = RGB(220, 40, 160),
    debugColor = RGB(220, 40, 160)
)
greenWallShader = Shader(
    reflectivity = 1,
    roughness = 0.68,
    color = RGB(20, 220, 220),
    debugColor = RGB(20, 220, 220)
)
box.setShader(boxShader)
box.setShader(boxEmissionShader, [48, 49])
box.setShader(redWallShader, [28, 29])
box.setShader(greenWallShader, [27, 26])
box.position = Vec3(0, 0, 1)

# monkey: Object = Object(scene=scene, name='Monkey', mesh=monkeyMesh)
# monkeyShader = Shader(
#     reflectivity = 0.2,
#     roughness = 0.04,
#     color = RGB(180, 180, 180),
#     debugColor = RGB(180, 180, 180)
# )
# monkeyMaterial = Material(
#     ior=1.4
# )
# monkey.setShader(monkeyShader)
# monkey.material = monkeyMaterial
# monkey.setTransforms()
# monkey.position = Vec3(1.44, -1.8, 1.4)
# monkey.scale = Scale(1.8, 1.8, 1.8)

cube: Cube = Cube(scene=scene, name='cube')
cubeShader = Shader(
    reflectivity = 0.15,
    roughness = 0.5,
    color = RGB(255, 255, 255),
    debugColor = RGB(0, 0, 0)
)
cubeMaterial = Material(
    ior=1.4
)
cube.setShader(cubeShader)
cube.material = cubeMaterial
cube.position = Vec3(1.44, -0.9, 1.4)
cube.scale = Scale(1.4, 2.6, 1.8)
cube.rotation.y = math.radians(10)

mirrorBall: Sphere = Sphere(scene=scene, name='Mirror ball')
mirrorBallShader = Shader(
    reflectivity = 1,
    roughness = 0,
    color = RGB(210, 210, 210),
    debugColor = RGB(180, 180, 180)
)
mirrorBall.setShader(mirrorBallShader)
mirrorBall.position = Vec3(-1.54, -2.1, 2)
mirrorBall.scale = Scale(3, 3, 3)

camera: Camera = Camera(scene=scene, name='Camera')
camera.length = 20
cameraShader = Shader(
    debugColor = RGB(0, 0, 0)
)
camera.setShader(cameraShader)
camera.position = Vec3(0, 0, -11)
# camera.position = Vec3(0, 0, -14)
# camera.position = Vec3(-3, 0, -11)
# camera.rotation.x = math.radians(4)

debug = Debug(scene=scene, options={
    'size': 3,
    'direction': 'side',
    'camera': True,
    'aabb': False,
    'bvh': False
})

renderer = Photon(scene=scene, options={
    'step': 20, ## TODO # Upscaling in post
    'fillStep': True,
    
    'bounces': 3,
    'samples': 2,
    
    'aabb': True,
    'bvh': True,
    
    'wait': False,
    'debug': False,
    'debugOptions': {
        'size': 4,
        'direction': 'side',
        'camera': True,
        'aabb': False,
        'bvh': False,
    },
    
    'lights': True,
    'emissionSampleDensity': 1,
    
    'ambient': RGB(32, 32, 32),
    
    'threads': 1,
})

tex = Texel(scene=scene)

sceneData = scene.toDict()

with open('cornellScene.json', 'w') as f:
    json.dump(sceneData, f)

# render = renderer.render()

# debug.inspect(render, scene)

# compressedRenderDict = Post.compressRenderedDict(render.toDict(), 10)

# data = compressedRenderDict
# with open('renders/renderedTest.json', 'w') as f:
#     json.dump(data, f)

# render = tex.render()

# img = render.imageAs('cv2')

# # debug.inspect(render, scene)

# cv2.imwrite(f'renders/cornell/highdef13|{render.time:.1f}|{render.rays}.png', img)

exit()