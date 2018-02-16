import json

with open('general.json') as data_file:
	data = json.load(data_file)
	latency_measures = []
	prev_user = data['messages'][0]['user']
	for i, message in enumerate(data['messages']):
		if 'user' in message:
			current_user = message['user']
			if current_user != prev_user:
				request_ts = message['ts']
				response_ts = data['messages'][i-1]['ts']
				latency_measure = float(response_ts) - float(request_ts)
				latency_measures += [latency_measure]
				prev_user = current_user
	latency_measures = sorted(latency_measures)
	length = len(latency_measures)
	first_quartile = length/4
	median = length/2
	third_quartile = first_quartile * 3
	print("first quartile is: ", latency_measures[first_quartile])
	print("median is: ", latency_measures[median])
	print("third quartile is: ", latency_measures[third_quartile])


