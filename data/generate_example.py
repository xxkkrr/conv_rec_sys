import csv

example_list = []
example_limit = 100
example_count = 0

with open('ratings_filter.csv') as f:
	f_csv = csv.reader(f)
	headers = next(f_csv)
	for row in f_csv:
		user_name = row[0]
		business_name = row[1]
		stars = float(row[2])

		if stars > 4.9:
			example_list.append(tuple([user_name, business_name, stars]))
			example_count += 1
			if example_count == example_limit:
				break

with open('example.csv','w') as f:
    f_csv = csv.writer(f)
    headers = ["user_id", "business_id", "stars"]
    f_csv.writerow(headers)
    f_csv.writerows(example_list)