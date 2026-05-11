import 'package:flutter/foundation.dart';

class VehicleProvider with ChangeNotifier {
  List<Map<String, dynamic>> _vehicles = [];
  bool _isLoading = false;
  String? _error;
  
  List<Map<String, dynamic>> get vehicles => _vehicles;
  bool get isLoading => _isLoading;
  String? get error => _error;
  
  void setVehicles(List<Map<String, dynamic>> vehicles) {
    _vehicles = vehicles;
    _error = null;
    notifyListeners();
  }
  
  void setLoading(bool loading) {
    _isLoading = loading;
    notifyListeners();
  }
  
  void setError(String error) {
    _error = error;
    notifyListeners();
  }
  
  void clearError() {
    _error = null;
    notifyListeners();
  }
}

