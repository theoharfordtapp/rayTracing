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
    emissionStrength = 0.6
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
box.setShader(boxEmissionShader, [24, 29])
box.setShader(redWallShader, [4, 5])
box.setShader(greenWallShader, [44, 45])
box.position = Vec3(0, 0, 1)

monkey: Object = Object(scene=scene, name='Monkey', mesh=monkeyMesh)
monkeyShader = Shader(
    reflectivity = 0.2,
    roughness = 0.04,
    color = RGB(180, 180, 180),
    debugColor = RGB(180, 180, 180)
)
monkeyMaterial = Material(
    ior=1.4
)
monkey.setShader(monkeyShader)
monkey.material = monkeyMaterial
monkey.setTransforms()
monkey.position = Vec3(1.44, -1.8, 1.4)
monkey.scale = Scale(1.8, 1.8, 1.8)

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
    'size': 4,
    'direction': 'side',
    'camera': True,
    'aabb': False,
    'bvh': False
})

renderer = Photon(scene=scene, options={
    'progressMode': 'none',
    
    'step': 80,
    'fillStep': True,
    
    'bounces': 3,
    'samples': 3,
    
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
    'emissionSampleDensity': 3,
    
    'ambient': RGB(32, 32, 32)
})

tex = Texel(scene=scene)

render = renderer.render()

debug.inspect(render, scene)

compressedRenderDict = Post.compressRenderedDict(render.toDict(), 10)

data = compressedRenderDict
with open('renders/renderedTest.json', 'w') as f:
    json.dump(data, f)

# FIXME # Texel face indexing only works after Photon has been run (on the same scene)

# render = tex.render()

# img = render.imageAs('cv2')

# debug.inspect(render, scene)

# cv2.imwrite(f'renders/cornell/highdef11|{render.time:.1f}|{render.rays}.png', img)

exit()