# SUPE Engine

## Installation

### Adding Scripts Path to PATH (Windows)

If you encounter a "command not found" error when running `pyengine`, add the following path to your system's PATH:

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
| `pyengine` or `pyengine help` | Show help message |
| `pyengine run` | Execute PyEngine code |
| `pyengine get` | Copy pyengine.py to current directory |

### Examples

1. Show help:
```bash
pyengine
```

2. Run PyEngine code:
```bash
pyengine run
```

Output:
```
Hello from pyengine!
This is the main pyengine code.
```

3. Copy pyengine.py file:
```bash
pyengine get
```

Output:
```
pyengine.py copied to: /current/directory/pyengine.py
```
