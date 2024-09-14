import 'package:flutter/material.dart';
import 'package:weatherapp/Worker/worker.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
class Loading extends StatefulWidget {
  const Loading({super.key});

  @override
  State<Loading> createState() => _LoadingState();
}

class _LoadingState extends State<Loading> {

  String city = "Dhaka";
  String? temp;
  String? humidity;
  String? air_speed;
  String? description;
  String? main;
  String? icon;

  void startApp(String city) async
  {
    worker instance = worker(location: city);
    await instance.getData();
    temp = instance.temp;
    humidity = instance.humidity;
    air_speed = instance.air_speed;
    description = instance.description;
    main = instance.main;
    icon = instance.icon;
    Future.delayed(const Duration(seconds: 2), () {
      Navigator.pushReplacementNamed(context, '/home', arguments:
      {"temp_value": temp,
        "hum_value": humidity,
        "air_value": air_speed,
        "des_value": description,
        "main_value": main,
        "icon_value": icon,
        "city_value": city,
      });
    });
  }

  //@override
  //void initState() {
  //  super.initState();
   // WidgetsBinding.instance.addPostFrameCallback((_) {
    //  startApp(city);
   // });
 // }

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      Map search = ModalRoute
          .of(context)
          ?.settings
          .arguments as Map? ?? {};
      if (search.isNotEmpty) {
        city = search["searchText"] ?? city;
      }
      startApp(city);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            const SizedBox(height: 100),
            Image.asset("images/wlogo.png", height: 240, width: 240),
            const SizedBox(height: 20),
            Text("Weather App",
              style: TextStyle(
                  fontSize: 30,
                  fontWeight: FontWeight.w500,
                  color: Colors.blue[800]
              ),
            ),
            const SizedBox(height: 15),
            Text("Made by Siyam",
              style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w400,
                  color: Colors.blue[800]
              ),
            ),
            const SizedBox(height: 50),
            const SpinKitWave(
              color: Colors.blueGrey,
              size: 50.0,
            ),
          ],
        ),
      ),
      backgroundColor: Colors.blue[100],
    );
  }
}