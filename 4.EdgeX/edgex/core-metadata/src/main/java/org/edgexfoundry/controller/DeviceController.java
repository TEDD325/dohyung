/*******************************************************************************
 * Copyright 2016-2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 * @microservice: core-metadata
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.controller;

import java.util.List;

import org.edgexfoundry.domain.meta.Device;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;

public interface DeviceController {

  /**
   * Fetch a specific device by database generated id. May return null if no device with the id is
   * found. Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) if device not found by id.
   * 
   * @param String database generated id for the device
   * 
   * @return device matching on the id
   */
  Device device(@PathVariable String id);

  /**
   * Return all devices sorted by id. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns LimitExceededException (HTTP 413) if the number returned exceeds
   * the max limit.
   * 
   * @return list of device
   */
  List<Device> devices();


  /**
   * Return Device matching given name (device names should be unique). May be null if no device
   * matches on the name provided. Returns ServiceException (HTTP 503) for unknown or unanticipated
   * issues. Returns NotFoundException (HTTP 404) if device not found by name.
   * 
   * @param name
   * @return device matching on name
   */
  Device deviceForName(@PathVariable String name);


  /**
   * Find all Devices having at least one label matching the label provided. List may be empty if no
   * device match. Returns ServiceException (HTTP 503) for unknown or unanticipated issues.
   * 
   * @param label - label to be matched
   * 
   * @return List of Device matching on specified label
   */
  List<Device> devicesByLabel(@PathVariable String label);

  /**
   * Find all devices associated to the DeviceService with the specified DeviceService database
   * generated identifier. List may be empty if no device match. Returns ServiceException (HTTP 503)
   * for unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if no DeviceService
   * match on the id provided.
   * 
   * @param serviceId - device service's database generated identifier
   * @return List of Devices associated to the device service
   */
  List<Device> devicesForService(@PathVariable String serviceId);

  /**
   * Find all devices associated to the DeviceService with the specified service name (DeviceService
   * names must be unique). List may be empty if no device match. Returns ServiceException (HTTP
   * 503) for unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if no
   * DeviceService match on the name provided.
   * 
   * @param servicename - device service's name
   * @return List of Devices associated to the device service
   */
  List<Device> devicesForServiceByName(@PathVariable String servicename);

  /**
   * Find all devices associated to the DeviceProfile with the specified profile database generated
   * identifier. List may be empty if no device match. Returns ServiceException (HTTP 503) for
   * unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if no DeviceProfile match
   * on the id provided.
   * 
   * @param profile id - device profile's database generated identifier
   * @return List of Devices associated to the device profile
   */
  List<Device> devicesForProfile(@PathVariable String profileId);

  /**
   * Find all devices associated to the DeviceProfile with the specified profile name. List may be
   * empty if no device match. Returns ServiceException (HTTP 503) for unknown or unanticipated
   * issues. Returns NotFoundException (HTTP 404) if no DeviceProfile match on the name provided.
   * 
   * @param profile name - device profile's name
   * @return List of Devices associated to the device profile
   */
  List<Device> devicesForProfileByName(@PathVariable String profilename);

  /**
   * Find all devices associated to the Addressable with the specified addressable database
   * generated identifier. List may be empty if no device match. Returns ServiceException (HTTP 503)
   * for unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if no Addressable
   * match on the id provided.
   * 
   * @param addressable id - addressable's database generated identifier
   * @return List of Devices associated to the addressable
   */
  List<Device> devicesForAddressable(@PathVariable String addressableId);

  /**
   * Find all devices associated to the Addressable with the specified addressable name. List may be
   * empty if no device match. Returns ServiceException (HTTP 503) for unknown or unanticipated
   * issues. Returns NotFoundException (HTTP 404) if no Addressable match on the name provided.
   * 
   * @param addressable name - addressable's name
   * @return List of Devices associated to the addressable
   */
  List<Device> devicesForAddressableByName(@PathVariable String addressablename);

  /**
   * Add a new Device - name must be unique. Embedded objects (device, service, profile,
   * addressable) are all referenced in the new Device object by id or name to associated objects.
   * All other data in the embedded objects will be ignored. Returns ServiceException (HTTP 503) for
   * unknown or unanticipated issues. Returns DataValidationException (HTTP 409) if an associated
   * object (Addressable, Profile, Service) cannot be found with the id or name provided.
   * 
   * @param Device object
   * @return database generated identifier for the new device
   */
  String add(@RequestBody Device device);

  /**
   * Update the last connected time of the device by database generated identifier. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if the device cannot be found by the identifier provided.
   * 
   * @param id - database generated identifier for the device
   * @param time - new last connected time in milliseconds
   * @return boolean indicating success of update
   */
  boolean updateLastConnected(@PathVariable String id, @PathVariable long time);

  /**
   * Update the last connected time of the device by database generated identifier. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if the device cannot be found by the identifier provided.
   * 
   * @param id - database generated identifier for the device
   * @param time - new last connected time in milliseconds
   * @param notify - boolean indicating whether callback should be made to device service to inform
   *        of update
   * @return boolean indicating success of update
   */
  boolean updateLastConnected(@PathVariable String id, @PathVariable long time,
      @PathVariable boolean notify);

  /**
   * Update the last connected time of the device by unique name of the device. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if the device cannot be found by the name provided.
   * 
   * @param name - device name
   * @param time - new last connected time in milliseconds
   * @return boolean indicating success of update
   */
  boolean updateLastConnectedByName(@PathVariable String name, @PathVariable long time);

  /**
   * Update the last connected time of the device by unique name of the device. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if the device cannot be found by the name provided.
   * 
   * @param name - device name
   * @param time - new last connected time in milliseconds
   * @param notify - boolean indicating whether callback should be made to device service to inform
   *        of update
   * 
   * @return boolean indicating success of update
   */
  boolean updateLastConnectedByName(@PathVariable String name, @PathVariable long time,
      @PathVariable boolean notify);

  /**
   * Update the last reported time of the device by database generated identifier. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if the device cannot be found by the identifier provided.
   * 
   * @param id - database generated identifier for the device
   * @param time - new last reported time in milliseconds
   * @return boolean indicating success of update
   */
  boolean updateLastReported(@PathVariable String id, @PathVariable long time);

  /**
   * Update the last reported time of the device by database generated identifier. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if the device cannot be found by the identifier provided.
   * 
   * @param id - database generated identifier for the device
   * @param time - new last reported time in milliseconds
   * @param notify - boolean indicating whether callback should be made to device service to inform
   *        of update
   * 
   * @return boolean indicating success of update
   */
  boolean updateLastReported(@PathVariable String id, @PathVariable long time,
      @PathVariable boolean notify);

  /**
   * Update the last reported time of the device by unique name of the device. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if the device cannot be found by the name provided.
   * 
   * @param name - device name
   * @param time - new last reported time in milliseconds
   * @return boolean indicating success of update
   */
  boolean updateLastReportedByName(@PathVariable String name, @PathVariable long time);

  /**
   * Update the last reported time of the device by unique name of the device. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if the device cannot be found by the name provided.
   * 
   * @param name - device name
   * @param time - new last reported time in milliseconds
   * @param notify - boolean indicating whether callback should be made to device service to inform
   *        of update
   * 
   * @return boolean indicating success of update
   */
  boolean updateLastReportedByName(@PathVariable String name, @PathVariable long time,
      @PathVariable boolean notify);

  /**
   * Update the op state of the device by database generated identifier. Returns ServiceException
   * (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if the
   * device cannot be found by the identifier provided. Returns DataValidationException (HTTP 409)
   * if the proposed new state is null;
   * 
   * @param id - database generated identifier for the device
   * @param opState - new op state for the device (either ENABLED or DISABLED)
   * @return - boolean indicating success of the operation
   */
  boolean updateOpState(@PathVariable String id, @PathVariable String opState);

  /**
   * Update the op status time of the device by unique name of the device. Returns ServiceException
   * (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if the
   * device cannot be found by the name provided. Returns DataValidationException (HTTP 409) if the
   * proposed new state is null;
   * 
   * @param name - device name
   * @param opState - new op state for the device (either ENABLED or DISABLED)
   * @return - boolean indicating success of the operation
   */
  boolean updateOpStateByName(@PathVariable String name, @PathVariable String opState);

  /**
   * Update the admin state of the device by database generated identifier. Returns ServiceException
   * (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if the
   * device cannot be found by the identifier provided. Returns DataValidationException (HTTP 409)
   * if the proposed new state is null;
   * 
   * @param id - database generated identifier for the device
   * @param adminstate - new admin state for the device (either LOCKED or UNLOCKED)
   * @return - boolean indicating success of the operation
   */
  boolean updateAdminState(@PathVariable String id, @PathVariable String adminState);

  /**
   * Update the admin state of the device by device name. Returns ServiceException (HTTP 503) for
   * unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if the device cannot be
   * found by the identifier provided. Returns DataValidationException (HTTP 409) if the proposed
   * new state is null;
   * 
   * @param name - device name
   * @param opState - new admin state for the device (either LOCKED or UNLOCKED)
   * @return - boolean indicating success of the operation
   */
  boolean updateAdminStateByName(@PathVariable String name, @PathVariable String adminState);

  /**
   * Update the Device identified by the id or name stored in the object provided. Id is used first,
   * name is used second for identification purposes. New device services & profiles cannot be
   * created with a PUT, but the service and profile can replaced by referring to a new device
   * service or profile id or name. Returns ServiceException (HTTP 503) for unknown or unanticipated
   * issues. Returns NotFoundException (HTTP 404) if the device cannot be found by the identifier
   * provided
   * 
   * @param device2 - object holding the identifier and new values for the Device
   * @return boolean indicating success of the update
   */
  boolean update(@RequestBody Device device2);

  /**
   * Remove the Device designated by database generated id. This does not remove associated objects
   * (addressable, service, profile, etc.). Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns NotFoundException (HTTP 404) if the device cannot be found by the
   * identifier provided.
   * 
   * @param database generated id for the device
   * @return boolean indicating success of the remove operation
   */
  boolean delete(@PathVariable String id);

  /**
   * Remove the Device designated by unique name. This does not remove associated objects
   * (addressable, service, profile, etc.). Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns NotFoundException (HTTP 404) if the device cannot be found by the
   * identifier provided.
   * 
   * @param unique name of the device
   * @return boolean indicating success of the remove operation
   */
  boolean deleteByName(@PathVariable String name);


}
