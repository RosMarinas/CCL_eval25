import json
d = json.load(open("submit_p14.json"))
for i in d:
    if i["idx"] in [213, 256, 287, 311, 317, 319, 320, 321, 322]:
        print(f'idx={i["idx"]}: choose_id={i["choose_id"]!r}')
print("Total entries:", len(d))
