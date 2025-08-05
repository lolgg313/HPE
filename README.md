![Hamid Py Engine](hpe.jpg)
# Hamid Py Engine ( HPE )

[![PyPI Downloads](https://static.pepy.tech/badge/hpe)](https://pepy.tech/projects/hpe)

![HPE](1.jpg) ![HPE](2.jpg)

## Installation

### Adding Scripts Path to PATH (Windows)

If you encounter a "command not found" error when running `hpe`, add the following path to your system's PATH:

1. Find your Python Scripts path:
   ```
   C:\Users\<YourName>\AppData\Roaming\Python\Python313\Scripts
   ```
   (Replace `<YourName>` with your actual username)

2. Add to PATH:
   - Press Windows + R
   - Type `sysdm.cpl` and press Enter
   - Go to the Advanced tab
   - Click Environment Variables
   - Under User variables, select Path and click Edit
   - Click New and paste the path above
   - Click OK on all windows

3. Close and reopen your terminal

## Usage

### Available Commands

| Command | Description |
|---------|-------------|
| `hpe` or `hpe help` | Show help message |
| `hpe run` | Execute hpe code |
| `hpe get` | Copy hpe.py to current directory |

### Examples

1. Show help:
```bash
hpe
```

2. Run hpe code:
```bash
hpe run
```

Output:
```
Hello from hpe!
This is the main hpe code.
```

3. Copy hpe.py file:
```bash
hpe get
```

Output:
```
hpe.py copied to: /current/directory/hpe.py
```
