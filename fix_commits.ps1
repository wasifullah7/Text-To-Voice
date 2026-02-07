# Fix git commits author and messages
git config user.name "wasifullah7"
git config user.email "wasif.wwez@gmail.com"

# Reset to the initial state (before any commits)
git update-ref -d HEAD

# Stage all files
git add .

# Create proper initial commit
git commit -m "Initial setup: Bark text-to-audio model with transformer architecture"

# Show result
git log --format='%H %an %ae %s'
