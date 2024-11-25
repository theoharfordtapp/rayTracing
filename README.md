# Python Ray-Tracing
### Theo HT

A simple ray-tracing engine built in Python from first principles, without the help of more complex libraries such as Numpy.

## Usage

After installing and importing the required libraries:

Create a `Scene()` instance. This represents the main 3D environment.

Create objects using instances of:\
`Object()` (an empty object which requires a mesh),\
`Cube()` (a premade cube mesh object),\
`Sphere()` (a mathematically-represented sphere, not using a mesh),\
`Light()` (a point-source light, emitting light in all directions—not visible to the camera,\
and *importantly*, `Camera()` (a camera object which is required to be able to render).

All of these objects require a `scene` parameter, which references the scene created earlier.\
They also take optional parameters such as `name`, `position`, `mesh`, `length`, `width`, `height`, `strength` (note that not all of these parameters apply to all types of object).

To create a mesh, either call the static method `Mesh.fromSTL()`, providing a `filepath` parameter; or create a `Mesh()` instance, providing `vertices` and `faces` parameters.

Objects have a `shader` property, which can be applied to a certain group of faces with `setShader(faceIndicesList)`. This list should be empty to apply the shader to the entire object.\
A shader is created using a `Shader()` instance, which has several parameters defining surface properties.\
An object can also have a `material` property, created with a `Material()` instance, for internal material properties such as refractive index.

Create a rendering engine. For ray-tracing, create a `Photon()` instance.\
For a primitive rasterisation (for a quick preview of a scene), create a `Texel()` instance.\
For an orthographic line-drawing projection (for debugging purposes etc.), create a `Debug()` instance.\
All three of these require a `scene` parameter, and take an optional `options` parameter (this will be merged with default options).

Render the scene using the `{engine}.render()` method on any rendering engine. This provides a `Rendered()` object, from which information such as `rays`, `time`, `traces`, etc. can be obtained.\
To get the actual image, use the `{rendered}.imageAs()`, providing a `type` parameter: `list` for raw image data, `np` for a numpy array, and `cv2` for a BGR-flipped numpy array, for use with Python-OpenCV.
