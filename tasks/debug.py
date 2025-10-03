import difflib

# Original string
original = """Hello, world!
This is a test file.
It contains some text."""

# Patch string (in unified diff format)
patch = """--- original.txt
+++ modified.txt
@@ -1,3 +1,3 @@
-Hello, world!
+Hello, patched world!
 This is a test file.
 It contains some text."""

# Split the patch into lines
patch_lines = patch.splitlines()

# Parse the patch into a list of diffs
diff = difflib.unified_diff(
    original.splitlines(),
    patch_lines,
)

# Apply the patch
result = difflib.(diff, original)

# Extract the patched string
patched_string = result[0]

print("Patched String:")
print(patched_string)