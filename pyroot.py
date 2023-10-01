import uproot
file = uproot.open("Hadr05.root")
print(file.keys())

print(file['1'].to_hist())
