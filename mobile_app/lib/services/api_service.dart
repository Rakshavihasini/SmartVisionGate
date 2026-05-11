import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ApiService {
  // CHANGE THIS TO YOUR API URL
  // For local testing: 'http://YOUR_COMPUTER_IP:5000'
  // For cloud: 'https://your-app.railway.app' or your Supabase function URL
  static const String baseUrl = 'BASE_URL'; // CHANGE THIS!
  
  Future<Map<String, dynamic>> registerVehicle({
    required File imageFile,
    required String licensePlate,
    String? ownerName,
    String? ownerPhone,
    String? ownerEmail,
    String? ownerAddress,
  }) async {
    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/api/register'),
      );
      
      // Add image
      request.files.add(
        await http.MultipartFile.fromPath('image', imageFile.path),
      );
      
      // Add fields
      request.fields['license_plate'] = licensePlate;
      if (ownerName != null && ownerName.isNotEmpty) {
        request.fields['owner_name'] = ownerName;
      }
      if (ownerPhone != null && ownerPhone.isNotEmpty) {
        request.fields['owner_phone'] = ownerPhone;
      }
      if (ownerEmail != null && ownerEmail.isNotEmpty) {
        request.fields['owner_email'] = ownerEmail;
      }
      if (ownerAddress != null && ownerAddress.isNotEmpty) {
        request.fields['owner_address'] = ownerAddress;
      }
      
      var response = await request.send();
      var responseBody = await response.stream.bytesToString();
      var jsonResponse = json.decode(responseBody);
      
      return jsonResponse;
    } catch (e) {
      return {
        'success': false,
        'error': 'CONNECTION_ERROR',
        'message': 'Failed to connect to server: $e',
      };
    }
  }

  Future<Map<String, dynamic>> verifyFace({
    required File imageFile,
  }) async {
    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/api/face/verify'),
      );

      request.files.add(
        await http.MultipartFile.fromPath('image', imageFile.path),
      );

      var response = await request.send();
      var responseBody = await response.stream.bytesToString();
      final jsonResponse = json.decode(responseBody);
      return jsonResponse;
    } catch (e) {
      return {
        'success': false,
        'authorized': false,
        'error': 'CONNECTION_ERROR',
        'message': 'Failed to connect to server: $e',
      };
    }
  }
  
  Future<Map<String, dynamic>> getVehicle(String licensePlate) async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/api/vehicle/$licensePlate'),
      );
      
      return json.decode(response.body);
    } catch (e) {
      return {
        'success': false,
        'error': 'CONNECTION_ERROR',
        'message': 'Failed to connect to server: $e',
      };
    }
  }
  
  Future<Map<String, dynamic>> listVehicles({String? search}) async {
    try {
      var url = '$baseUrl/api/vehicles';
      if (search != null && search.isNotEmpty) {
        url += '?search=$search';
      }
      
      final response = await http.get(Uri.parse(url));
      return json.decode(response.body);
    } catch (e) {
      return {
        'success': false,
        'error': 'CONNECTION_ERROR',
        'message': 'Failed to connect to server: $e',
      };
    }
  }
  
  Future<Map<String, dynamic>> getStats() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/api/stats'));
      return json.decode(response.body);
    } catch (e) {
      return {
        'success': false,
        'error': 'CONNECTION_ERROR',
        'message': 'Failed to connect to server: $e',
      };
    }
  }
  
  Future<bool> checkHealth() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/health'));
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }
}

