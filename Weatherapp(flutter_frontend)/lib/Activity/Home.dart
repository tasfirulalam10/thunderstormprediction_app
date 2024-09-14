import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter_gradient_app_bar/flutter_gradient_app_bar.dart';
import 'package:weather_icons/weather_icons.dart';
import 'thunderstorm_forecasting_page.dart'; // Import the new page

class Home extends StatefulWidget {
  const Home({super.key});

  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  TextEditingController searchController = TextEditingController();

  @override
  void initState() {
    super.initState();
    print("This is a init state");
  }

  @override
  void setState(VoidCallback fn) {
    super.setState(fn);
    print("Set State Called");
  }

  @override
  void dispose() {
    super.dispose();
    print("Widget Disposed");
  }

  @override
  Widget build(BuildContext context) {
    Map info = ModalRoute.of(context)?.settings.arguments as Map? ?? {};

    var cityName = [
      "Dhaka",
      "Chittagong",
      "Mymensingh",
      "Rajshahi",
      "Khulna",
      "Barishal",
      "Rangpur",
      "Sylhet"
    ];
    final random = Random();
    var city = "";
    if (cityName.isNotEmpty) {
      city = cityName[random.nextInt(cityName.length)];
    } else {
      print("Error: cityName list is empty.");
    }

    String icon = (info['icon_value'] ?? 'default_icon').toString();
    String gcity = info['city_value'] ?? 'Unknown City';
    String hum = info['hum_value'] ?? 'N/A';
    String des = info['des_value'] ?? 'No description';
    String temp = (info['temp_value']?.toString() ?? 'N/A').split('.')[0];
    String air = (info['air_value']?.toString() ?? 'N/A').split('.')[0];

    print(icon);
    return Scaffold(
      resizeToAvoidBottomInset: false,
      appBar: PreferredSize(
          preferredSize: const Size.fromHeight(0),
          child: GradientAppBar(
            gradient: LinearGradient(
                colors: [Colors.white, Colors.blueAccent],
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter),
          )),
      body: SafeArea(
        child: SingleChildScrollView(
          child: Container(
            decoration: const BoxDecoration(
                gradient: LinearGradient(
                    begin: Alignment.topCenter,
                    end: Alignment.centerRight,
                    colors: [Colors.lightBlueAccent, Colors.lightBlueAccent])),
            child: Column(
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8),
                  margin: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
                  decoration: BoxDecoration(
                      color: Colors.white70,
                      borderRadius: BorderRadius.circular(25)),
                  child: Row(
                    children: [
                      GestureDetector(
                        onTap: () {
                          Navigator.pushReplacementNamed(context, "/Loading", arguments: {
                            "searchText": searchController.text,
                          });
                        },
                        child: Container(
                          margin: const EdgeInsets.fromLTRB(3, 0, 7, 0),
                          child: const Icon(
                            Icons.search,
                            color: Colors.blue,
                          ),
                        ),
                      ),
                      Expanded(
                        child: TextField(
                          controller: searchController,
                          decoration: InputDecoration(
                              border: InputBorder.none,
                              hintText: "Search $city"),
                        ),
                      ),
                    ],
                  ),
                ),
                Row(
                  children: [
                    Expanded(
                      flex: 1,
                      child: Container(
                          decoration: BoxDecoration(
                              borderRadius: BorderRadius.circular(14),
                              color: Colors.white.withOpacity(0.5)),
                          margin: const EdgeInsets.symmetric(horizontal: 25),
                          padding: const EdgeInsets.fromLTRB(10, 10, 10, 10),
                          child: Wrap(
                            crossAxisAlignment: WrapCrossAlignment.start,
                            children: [
                              Image.network("https://openweathermap.org/img/wn/$icon@2x.png"),
                              SizedBox(width: 50),
                              Column(
                                children: [
                                  Text(
                                    des,
                                    style: TextStyle(
                                        fontSize: 25,
                                        fontWeight: FontWeight.bold),
                                  ),
                                  Text(
                                    " In $gcity",
                                    style: TextStyle(
                                        fontSize: 25,
                                        fontWeight: FontWeight.bold),
                                  )
                                ],
                              )
                            ],
                          )),
                    ),
                  ],
                ),
                Row(
                  children: [
                    Expanded(
                      child: Container(
                          height: 200,
                          decoration: BoxDecoration(
                              borderRadius: BorderRadius.circular(14),
                              color: Colors.white.withOpacity(0.5)),
                          margin: const EdgeInsets.symmetric(
                              horizontal: 25, vertical: 10),
                          padding: const EdgeInsets.all(26),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Icon(WeatherIcons.thermometer),
                              Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Text(
                                    temp,
                                    style: TextStyle(
                                      fontSize: 70,
                                    ),
                                  ),
                                  Text(
                                    "C",
                                    style: TextStyle(
                                        fontSize: 30,
                                        fontWeight: FontWeight.bold),
                                  )
                                ],
                              )
                            ],
                          )),
                    ),
                  ],
                ),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    Expanded(
                      child: Container(
                          decoration: BoxDecoration(
                              borderRadius: BorderRadius.circular(14),
                              color: Colors.white.withOpacity(0.5)),
                          margin: const EdgeInsets.fromLTRB(20, 0, 10, 0),
                          padding: const EdgeInsets.all(15),
                          height: 150,
                          child: Column(
                            children: [
                              Row(
                                mainAxisAlignment: MainAxisAlignment.start,
                                children: [
                                  Icon(WeatherIcons.windy),
                                ],
                              ),
                              Text(
                                air,
                                style: TextStyle(
                                  fontSize: 40,
                                ),
                              ),
                              Text("Km/hr")
                            ],
                          )),
                    ),
                    Expanded(
                      child: Container(
                          decoration: BoxDecoration(
                              borderRadius: BorderRadius.circular(14),
                              color: Colors.white.withOpacity(0.5)),
                          margin: const EdgeInsets.fromLTRB(10, 0, 20, 0),
                          padding: const EdgeInsets.all(15),
                          height: 150,
                          child: Column(
                            children: [
                              Row(
                                mainAxisAlignment: MainAxisAlignment.start,
                                children: [
                                  Icon(WeatherIcons.humidity),
                                ],
                              ),
                              Text(
                                hum,
                                style: TextStyle(
                                  fontSize: 40,
                                ),
                              ),
                              Text("%")
                            ],
                          )),
                    ),
                  ],
                ),
                SizedBox(height: 20),
                ElevatedButton(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => ThunderstormForecastingPage(),
                      ),
                    );
                  },
                  child: const Text("Show Thunderstorm Forecasting"),
                ),

                SizedBox(height: 20),
                Container(
                  padding: const EdgeInsets.all(5),
                  child: const Column(
                    mainAxisAlignment: MainAxisAlignment.end,
                    children: [
                      Text("Made by Siyam"),
                      Text("Data provided by openweatherapp.org"),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
