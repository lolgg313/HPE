# ![logo](hpe.jpg) Hamid Py Engine (HPE)

[![PyPI Version](https://img.shields.io/pypi/v/hpe.svg)](https://pypi.org/project/hpe)  
[![Python Versions](https://img.shields.io/pypi/pyversions/hpe.svg)](https://pypi.org/project/hpe)  
[![PyPI Downloads](https://static.pepy.tech/badge/hpe)](https://pepy.tech/project/hpe)  
[![PyPI Downloads Monthly](https://static.pepy.tech/badge/hpe/month)](https://pepy.tech/project/hpe)  
[![License](https://img.shields.io/github/license/lolgg313/HPE)](./LICENSE)  
[![Last Commit](https://img.shields.io/github/last-commit/lolgg313/HPE)](https://github.com/lolgg313/HPE)  
[![Repo Size](https://img.shields.io/github/repo-size/lolgg313/HPE)](https://github.com/lolgg313/HPE)  
[![Issues](https://img.shields.io/github/issues/lolgg313/HPE)](https://github.com/lolgg313/HPE/issues)  
[![Forks](https://img.shields.io/github/forks/lolgg313/HPE?style=social)](https://github.com/lolgg313/HPE/network)  
[![Stars](https://img.shields.io/github/stars/lolgg313/HPE?style=social)](https://github.com/lolgg313/HPE/stargazers)

> ✨ A lightweight 3D Game engine purely made in python!

## About HPE: The High-Performance Python Engine for 3D World Creation

HPE is a powerful, feature-rich 3D game engine built entirely in Python. It's designed for creators and developers looking to craft ambitious 3D worlds with the simplicity and speed of Python. HPE bridges the gap between simple rendering libraries and complex professional software by providing a complete, Unity-style visual editor, allowing you to build, test, and iterate on your games in real-time.

From dynamic skies to sculptable terrain, HPE provides the tools to bring your vision to life.

---

### Key Capabilities 🚀

* 🧑‍💻 Integrated Visual Editor: A full suite of tools at your fingertips, including a scene hierarchy, a real-time properties inspector for selected objects, a console, and interactive 3D transformation gizmos (translate/rotate).

* 🎨 Advanced Rendering Pipeline:
    * Procedural Sky & Animated Clouds: Generate beautiful, dynamic skies with real-time animated clouds and customizable sun, halo, and fog colors.
    * Modern Model Support: Seamlessly import industry-standard GLTF and GLB models into your scene.
    * High-Quality Visuals: Features a robust rendering pipeline with dynamic lighting, shadow effects, and HBAO-like ambient occlusion for creating immersive environments.

* 🌍 Powerful World-Building Tools:
    * Real-Time Terrain Editor: Create vast landscapes and sculpt them directly within the editor.
    * Intuitive Sculpting: Raise, lower, and smooth your terrain with an adjustable brush size and strength.
    * 3D Primitives & Enemies: Instantly add and manipulate basic shapes like cubes, spheres, and capsules, or spawn pre-configured enemies for your game.

* ⚙️ Physics and Interactivity:
    * Built-in Physics Engine: Simulate realistic interactions with support for Static and RigidBody physics.
    * FPS Controller Mode: Jump directly into your game with a "Play" button that activates a first-person controller for in-engine testing.
    * Scripting Support: Attach custom .blob script files to any object to define unique behaviors and game logic.

* 🔧 Efficient and Extensible Workflow:
    * Full Scene Serialization: Save and load your entire scenes, including all objects, components, terrain heightmaps, and environment settings, with the portable .hamidmap format.
    * Performance Optimized: Leverages Numba for JIT-compiled physics calculations and multithreading for smooth asset loading and processing.

Whether you're prototyping a new game idea, building stunningly detailed environments, or learning the fundamentals of 3D development, HPE provides a powerful and intuitive platform. Dive into the future of Python game development and start creating your world today!

> [!Note]
> `Fixed` TERRAIN EDITOR ( TERRAIN EDITOR had been added with more set of tools. )

# Minimum System
> Cpu core i5 8th gen
> 
> Ram: 8 gb
> 
> Graphic Card: 8 gb vram
> 
> Memory Needed: 512 MB ~ 700 MB

---

![HPE Screenshot 1](1.jpg)
![HPE Screenshot 2](2.jpg)

---

## 📦 Installation

```bash
pip install hpe
```

---

## ⚙️ Add Scripts Path to `PATH` (Windows)

If the `hpe` command is not recognized:

1. Locate your Scripts path:
   ```
   C:\Users\<YourName>\AppData\Roaming\Python\Python313\Scripts
   ```
   Replace `<YourName>` with your actual Windows username.

2. Add to PATH:
   - Press **Win + R**, run `sysdm.cpl`
   - Go to **Advanced** tab > **Environment Variables**
   - Under *User variables*, select **Path** > **Edit** > **New** > paste the path
   - Click OK on all windows and restart your terminal

---

## 🚀 Usage

| Command         | Description                          |
|----------------|--------------------------------------|
| `hpe` / `hpe help` | Show help message                 |
| `hpe run`       | Run the main HPE code                |
| `hpe get`       | Copy `hpe.py` to current directory   |

---

### 🧪 Examples

```bash
hpe
```

```bash
hpe run
```

**Output:**
```
Hello from hpe!
This is the main hpe code.
```

```bash
hpe get
```

**Output:**
```
hpe.py copied to: /current/directory/hpe.py
```
