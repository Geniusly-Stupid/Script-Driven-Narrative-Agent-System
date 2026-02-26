import sqlite3
conn = sqlite3.connect("narrative.db")
conn.row_factory = sqlite3.Row
cur = conn.cursor()

print("Scenes:")
for r in cur.execute("SELECT scene_id, scene_goal, status FROM scenes"):
    print(dict(r))

print("\nPlots:")
for r in cur.execute("SELECT plot_id, scene_id, plot_goal, status, progress FROM plots"):
    print(dict(r))
