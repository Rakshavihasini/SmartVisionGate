import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:intl/intl.dart';
import '../services/api_service.dart';
import '../providers/vehicle_provider.dart';

class RecordsScreen extends StatefulWidget {
  const RecordsScreen({super.key});

  @override
  State<RecordsScreen> createState() => _RecordsScreenState();
}

class _RecordsScreenState extends State<RecordsScreen> {
  final _searchController = TextEditingController();
  bool _isLoading = false;
  
  @override
  void initState() {
    super.initState();
    _loadVehicles();
  }
  
  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }
  
  Future<void> _loadVehicles({String? search}) async {
    setState(() {
      _isLoading = true;
    });
    
    try {
      final apiService = context.read<ApiService>();
      final result = await apiService.listVehicles(search: search);
      
      if (mounted) {
        if (result['success'] == true) {
          context.read<VehicleProvider>().setVehicles(
            List<Map<String, dynamic>>.from(result['vehicles'] ?? []),
          );
        } else {
          context.read<VehicleProvider>().setError(result['message'] ?? 'Failed to load vehicles');
        }
      }
    } catch (e) {
      if (mounted) {
        context.read<VehicleProvider>().setError('Error: $e');
      }
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }
  
  void _onSearchChanged() {
    _loadVehicles(search: _searchController.text);
  }
  
  String _formatDate(String? dateStr) {
    if (dateStr == null) return 'N/A';
    try {
      final date = DateTime.parse(dateStr);
      return DateFormat('MMM dd, yyyy').format(date);
    } catch (e) {
      return dateStr;
    }
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Vehicle Records'),
        backgroundColor: const Color(0xFF2563EB),
        foregroundColor: Colors.white,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadVehicles,
          ),
        ],
      ),
      body: Column(
        children: [
          // Search Bar
          Padding(
            padding: const EdgeInsets.all(16),
            child: TextField(
              controller: _searchController,
              decoration: InputDecoration(
                hintText: 'Search by license plate, owner, make...',
                prefixIcon: const Icon(Icons.search),
                suffixIcon: _searchController.text.isNotEmpty
                    ? IconButton(
                        icon: const Icon(Icons.clear),
                        onPressed: () {
                          _searchController.clear();
                          _onSearchChanged();
                        },
                      )
                    : null,
              ),
              onChanged: (_) => _onSearchChanged(),
            ),
          ),
          
          // Vehicle List
          Expanded(
            child: Consumer<VehicleProvider>(
              builder: (context, provider, child) {
                if (_isLoading && provider.vehicles.isEmpty) {
                  return const Center(
                    child: CircularProgressIndicator(),
                  );
                }
                
                if (provider.error != null) {
                  return Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const Icon(
                          Icons.error_outline,
                          size: 64,
                          color: Colors.red,
                        ),
                        const SizedBox(height: 16),
                        Text(
                          provider.error!,
                          textAlign: TextAlign.center,
                          style: const TextStyle(color: Colors.red),
                        ),
                        const SizedBox(height: 16),
                        ElevatedButton.icon(
                          onPressed: _loadVehicles,
                          icon: const Icon(Icons.refresh),
                          label: const Text('Retry'),
                        ),
                      ],
                    ),
                  );
                }
                
                if (provider.vehicles.isEmpty) {
                  return Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          Icons.folder_open,
                          size: 80,
                          color: Colors.grey[300],
                        ),
                        const SizedBox(height: 16),
                        Text(
                          'No vehicles registered yet',
                          style: TextStyle(
                            fontSize: 18,
                            color: Colors.grey[600],
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Register your first vehicle to get started',
                          style: TextStyle(
                            color: Colors.grey[500],
                          ),
                        ),
                      ],
                    ),
                  );
                }
                
                return RefreshIndicator(
                  onRefresh: () => _loadVehicles(),
                  child: ListView.builder(
                    padding: const EdgeInsets.all(16),
                    itemCount: provider.vehicles.length,
                    itemBuilder: (context, index) {
                      final vehicle = provider.vehicles[index];
                      return _VehicleCard(
                        vehicle: vehicle,
                        formatDate: _formatDate,
                      );
                    },
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}

class _VehicleCard extends StatelessWidget {
  final Map<String, dynamic> vehicle;
  final String Function(String?) formatDate;
  
  const _VehicleCard({
    required this.vehicle,
    required this.formatDate,
  });
  
  @override
  Widget build(BuildContext context) {
    final licensePlate = vehicle['license_plate'] ?? 'N/A';
    final ownerName = vehicle['owner_name'];
    final ownerPhone = vehicle['owner_phone'];
    final ownerEmail = vehicle['owner_email'];
    final vehicleType = vehicle['vehicle_type'];
    final vehicleColor = vehicle['vehicle_color'];
    final vehicleMake = vehicle['vehicle_make'];
    final vehicleModel = vehicle['vehicle_model'];
    final vehicleYear = vehicle['vehicle_year'];
    final createdAt = vehicle['created_at'];
    
    return Card(
      margin: const EdgeInsets.only(bottom: 16),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header
            Row(
              children: [
                Container(
                  width: 56,
                  height: 56,
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(
                      colors: [Color(0xFF2563EB), Color(0xFF1e40af)],
                    ),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Icon(
                    Icons.directions_car,
                    color: Colors.white,
                    size: 32,
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        licensePlate,
                        style: const TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      if (vehicleMake != null || vehicleModel != null)
                        Text(
                          '${vehicleMake ?? ''} ${vehicleModel ?? ''}'.trim(),
                          style: TextStyle(
                            color: Colors.grey[600],
                            fontSize: 14,
                          ),
                        ),
                    ],
                  ),
                ),
              ],
            ),
            
            const SizedBox(height: 16),
            const Divider(),
            const SizedBox(height: 8),
            
            // Details
            if (ownerName != null && ownerName.isNotEmpty)
              _InfoRow(
                icon: Icons.person,
                label: 'Owner',
                value: ownerName,
              ),
            
            if (ownerPhone != null && ownerPhone.isNotEmpty)
              _InfoRow(
                icon: Icons.phone,
                label: 'Phone',
                value: ownerPhone,
              ),
            
            if (ownerEmail != null && ownerEmail.isNotEmpty)
              _InfoRow(
                icon: Icons.email,
                label: 'Email',
                value: ownerEmail,
              ),
            
            if (vehicleType != null && vehicleType.isNotEmpty)
              _InfoRow(
                icon: Icons.category,
                label: 'Type',
                value: vehicleType,
              ),
            
            if (vehicleColor != null && vehicleColor.isNotEmpty)
              _InfoRow(
                icon: Icons.color_lens,
                label: 'Color',
                value: vehicleColor,
              ),
            
            if (vehicleYear != null && vehicleYear.isNotEmpty)
              _InfoRow(
                icon: Icons.calendar_today,
                label: 'Year',
                value: vehicleYear,
              ),
            
            if (createdAt != null)
              _InfoRow(
                icon: Icons.access_time,
                label: 'Registered',
                value: formatDate(createdAt),
              ),
          ],
        ),
      ),
    );
  }
}

class _InfoRow extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  
  const _InfoRow({
    required this.icon,
    required this.label,
    required this.value,
  });
  
  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Icon(
            icon,
            size: 18,
            color: const Color(0xFF2563EB),
          ),
          const SizedBox(width: 8),
          Text(
            '$label:',
            style: const TextStyle(
              fontWeight: FontWeight.w600,
              fontSize: 14,
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              value,
              style: TextStyle(
                color: Colors.grey[700],
                fontSize: 14,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

