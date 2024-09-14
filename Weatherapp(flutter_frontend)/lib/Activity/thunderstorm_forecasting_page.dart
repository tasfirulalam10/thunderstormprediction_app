import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(ThunderstormForecastingApp());
}

class ThunderstormForecastingApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Thunderstorm Forecasting',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: ThunderstormForecastingPage(),
    );
  }
}

class ThunderstormForecastingPage extends StatefulWidget {
  @override
  _ThunderstormForecastingPageState createState() => _ThunderstormForecastingPageState();
}

class _ThunderstormForecastingPageState extends State<ThunderstormForecastingPage> {
  final _districtController = TextEditingController();
  final _dateController = TextEditingController();
  String predictedValue = '';

  Future<List<Map<String, num>>> fetchForecast(String district, String date) async {
    final response = await http.get(Uri.parse('http://10.19.89.195:4000/forecast?district=$district&date=$date'));

    print('Forecast API Response Status: ${response.statusCode}');
    print('Forecast API Response Body: ${response.body}');

    if (response.statusCode == 200) {
      List<dynamic> decodedBody = json.decode(response.body);
      List<Map<String, num>> forecastList = decodedBody.map((item) => Map<String, num>.from(item)).toList();
      return forecastList;
    } else {
      throw Exception('Failed to load forecast');
    }
  }

  Future<String> sendForecastToFlask(List<Map<String, num>> forecastList) async {
    print("Sending forecast to Flask: $forecastList");
    final response = await http.post(
      Uri.parse('http://10.0.2.2:5000/predict'),
      headers: <String, String>{
        'Content-Type': 'application/json; charset=UTF-8',
      },
      body: jsonEncode(forecastList),
    );

    print('Flask API Response Status: ${response.statusCode}');
    print('Flask API Response Body: ${response.body}');

    if (response.statusCode == 200) {
      Map<String, dynamic> data = json.decode(response.body);
      return data['predicted_value'].toString();
    } else {
      throw Exception('Failed to send data to Flask');
    }
  }

  void _handleSubmit() async {
    final district = _districtController.text;
    final date = _dateController.text;

    if (!RegExp(r'^\d{4}-\d{2}-\d{2}$').hasMatch(date)) {
      try {
        print("Fetching forecast for district: $district, date: $date");
        final forecastList = await fetchForecast(district, date);
        print("Forecast fetched successfully: $forecastList");
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Data sent to Model')));
        String value = await sendForecastToFlask(forecastList);
        setState(() {
          predictedValue = value;
        });
      } catch (e) {
        print("Error: $e");
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Failed to fetch/send data: $e')));
      }
    } else {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Invalid date format. Use YYYY-MM-DD.')));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Thunderstorm Forecasting'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextField(
              controller: _districtController,
              decoration: InputDecoration(labelText: 'District'),
            ),
            TextField(
              controller: _dateController,
              decoration: InputDecoration(labelText: 'Date (MM/DD/YYYY)'),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: _handleSubmit,
              child: Text('Get The Prediction'),
            ),
            SizedBox(height: 20),
            if (predictedValue.isNotEmpty)
              Builder(
                builder: (context) {
                  int value = 0;
                  try {
                    value = int.parse(predictedValue);
                  } catch (e) {
                    print('Error parsing predicted value: $e');
                  }
                  if (value > 10) {
                    return Column(
                      children: [
                        Text(
                          'Predicted Value: $predictedValue\nThere is a probability of thunderstorm',
                          style: TextStyle(fontSize: 20, color: Colors.red),
                        ),
                        SizedBox(height: 20),
                        Text(
                          'Necessary steps to follow:',
                          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                        ),
                        ListView(
                          shrinkWrap: true,
                          children: [
                            ListTile(
                              leading: Icon(Icons.check),
                              title: Text('Stay indoors and avoid travel if possible.'),
                            ),
                            ListTile(
                              leading: Icon(Icons.check),
                              title: Text('Secure outdoor objects that can be blown away.'),
                            ),
                            ListTile(
                              leading: Icon(Icons.check),
                              title: Text('Unplug electronic devices to prevent damage.'),
                            ),
                            ListTile(
                              leading: Icon(Icons.check),
                              title: Text('Stay informed about weather updates.'),
                            ),
                          ],
                        ),
                      ],
                    );
                  } else {
                    return Text(
                      'Predicted Value: 0\nThere will be no thunderstorm on this date.',
                      style: TextStyle(fontSize: 20, color: Colors.green),
                    );
                  }
                },
              ),
          ],
        ),
      ),
    );
  }
}