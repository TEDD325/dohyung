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
import java.util.Set;

import org.edgexfoundry.domain.meta.Addressable;
import org.edgexfoundry.domain.meta.DeviceService;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;

public interface DeviceServiceController {

  /**
   * Fetch a specific device service by database generated id. May return null if no service with
   * the id is found. Returns ServiceException (HTTP 503) for unknown or unanticipated issues.
   * Returns NotFoundException (HTTP 404) if the device service is not found by the id provided.
   * 
   * @param String database generated id for the service
   * 
   * @return device service matching on id
   */
  DeviceService deviceService(@PathVariable String id);

  /**
   * Return the DeviceService matching given name (service names should be unique). May be null if
   * no services matches on the name provided. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns NotFoundException (HTTP 404) if the device service is not found
   * by the name provided.
   * 
   * @param name
   * @return device service matching on name
   */
  DeviceService deviceServiceForName(@PathVariable String name);

  /**
   * Return all device services sorted by id. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns LimitExceededException (HTTP 413) if the number returned exceeds
   * the max limit.
   * 
   * @return list of profiles
   */
  List<DeviceService> deviceServices();

  /**
   * Find all device servicess associated to the Addressable with the specified addressable database
   * generated identifier. List may be empty if no device service match. Returns ServiceException
   * (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if no
   * Addressable match on the id provided.
   * 
   * @param addressable id - addressable's database generated identifier
   * @return List of DeviceServices associated to the addressable
   */
  List<DeviceService> deviceServicesForAddressable(@PathVariable String addressableId);

  /**
   * Find all device serices associated to the Addressable with the specified addressable name. List
   * may be empty if no device services match. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns NotFoundException (HTTP 404) if no Addressable match on the name
   * provided.
   * 
   * @param addressable name - addressable's name
   * @return List of DeviceServices associated to the addressable
   */
  List<DeviceService> deviceServicesForAddressableByName(@PathVariable String addressablename);

  /**
   * Find all DeviceServices having at least one label matching the label provided. List may be
   * empty if no device services match. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues.
   * 
   * @param label - label to be matched
   * 
   * @return List of DeviceService matching on specified label
   */
  List<DeviceService> deviceServicesByLabel(@PathVariable String label);

  /**
   * Return a set (set versus list to insure element uniqueness) of addressables that are associated
   * to the devices of a device service. The device service is identified by id. Returns
   * NotFoundException (HTTP 404) if the device service is not found by id. Returns ServiceException
   * (HTTP 503) for unknown or unanticipated issues.
   * 
   * @param id - database provided id for the device service
   * @return - set of unique addressable objects associated to devices of the owning device service
   */
  Set<Addressable> addressablesForAssociatedDevices(@PathVariable String id);

  /**
   * Return a set (set versus list to insure element uniqueness) of addressables that are associated
   * to the devices of a device service. The device service is identified by name. Returns
   * NotFoundException (HTTP 404) if the device service is not found by id. Returns ServiceException
   * (HTTP 503) for unknown or unanticipated issues.
   * 
   * @param name - unique name for the device service
   * @return - set of unique addressable objects associated to devices of the owning device service
   */
  Set<Addressable> addressablesForAssociatedDevicesByName(@PathVariable String name);

  /**
   * Add a new DeviceService - name must be unique. Returns ServiceException (HTTP 503) for unknown
   * or unanticipated issues. Returns DataValidationException (HTTP 409) if an associated
   * addressable (by id or name) is not found.
   * 
   * @param DeviceService object
   * @return database generated identifier for the new device service
   */
  String add(@RequestBody DeviceService deviceService);

  /**
   * Update the last connected time of the device service by database generated identifier. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if the device service cannot be found by the identifier provided.
   * 
   * @param id - database generated identifier for the device service
   * @param time - new last connected time in milliseconds
   * @return boolean indicating success of update
   */
  boolean updateLastConnected(@PathVariable String id, @PathVariable long time);

  /**
   * Update the last connected time of the device service by unique name of the device service.
   * Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) if the device service cannot be found by the name provided.
   * 
   * @param name - device service name
   * @param time - new last connected time in milliseconds
   * @return boolean indicating success of update
   */
  boolean updateLastConnectedByName(@PathVariable String name, @PathVariable long time);

  /**
   * Update the last reported time of the device service by database generated identifier. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if the device service cannot be found by the identifier provided.
   * 
   * @param id - database generated identifier for the device service
   * @param time - new last reported time in milliseconds
   * @return boolean indicating success of update
   */
  boolean updateLastReported(@PathVariable String id, @PathVariable long time);

  /**
   * Update the last reported time of the device service by unique name of the device service.
   * Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) if the device service cannot be found by the name provided.
   * 
   * @param name - device service name
   * @param time - new last reported time in milliseconds
   * @return boolean indicating success of update
   */
  boolean updateLastReportedByName(@PathVariable String name, @PathVariable long time);

  /**
   * Update the op state of the device service by database generated identifier. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if the device service cannot be found by the identifier provided.
   * 
   * @param id - database generated identifier for the device service
   * @param opState - new op state for the device service (either ENABLED or DISABLED)
   * @return - boolean indicating success of the operation
   */
  boolean updateOpState(@PathVariable String id, @PathVariable String opState);

  /**
   * Update the op status time of the device service by unique name of the device service. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if the device service cannot be found by the name provided.
   * 
   * @param name - device service name
   * @param opState - new op state for the device service (either ENABLED or DISABLED)
   * @return - boolean indicating success of the operation
   */
  boolean updateOpStateByName(@PathVariable String name, @PathVariable String opState);

  /**
   * Update the admin state of the device service by database generated identifier. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if the device service cannot be found by the identifier provided.
   * 
   * @param id - database generated identifier for the device service
   * @param adminstate - new admin state for the device service (either LOCKED or UNLOCKED)
   * @return - boolean indicating success of the operation
   */
  boolean updateAdminState(@PathVariable String id, @PathVariable String adminState);

  /**
   * Update the admin state of the device service by device service name. Returns ServiceException
   * (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if the
   * device service cannot be found by the identifier provided.
   * 
   * @param name - device service name
   * @param opState - new admin state for the device service (either LOCKED or UNLOCKED)
   * @return - boolean indicating success of the operation
   */
  boolean updateAdminStateByName(@PathVariable String name, @PathVariable String adminState);

  /**
   * Update the DeviceServcie identified by the id or name stored in the object provided. Id is used
   * first, name is used second for identification purposes. Returns ServiceException (HTTP 503) for
   * unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if the device service
   * cannot be found by the identifier provided.
   * 
   * @param deviceService2 - object holding the identifier and new values for the DeviceService
   * @return boolean indicating success of the update
   */
  boolean update(@RequestBody DeviceService deviceService2);

  /**
   * Remove the DeviceService designated by database generated id. Returns ServiceException (HTTP
   * 503) for unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if the device
   * service cannot be found by the identifier provided.
   * 
   * 
   * @param database generated id for the device service
   * @return boolean indicating success of the remove operation
   */
  boolean delete(@PathVariable String id);

  /**
   * Remove the DeviceService designated by name. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns NotFoundException (HTTP 404) if the device service cannot be
   * found by the name provided.
   * 
   * 
   * @param name for the device service
   * @return boolean indicating success of the remove operation
   */
  boolean deleteByName(@PathVariable String name);

}
