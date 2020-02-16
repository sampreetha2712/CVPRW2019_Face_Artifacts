import csv

files = ['2008_000003.jpg', 'demo_fake.mp4', 'demo.mp4']
prob = [-1, 0.99737376, 0.01950417]
label_prob = []
for p in prob:
    if(p == -1):
        label_prob.append("REAL")
    elif(p < 0.5):
        label_prob.append("REAL")
    else:
        label_prob.append("FAKE")
print(label_prob)
result = list(map(list, zip(files, label_prob)))
print(result)
with open('result.csv', 'w+') as file:
    writer = csv.writer(file)
    writer.writerow(["file name", "label"])
    writer.writerows(result)
