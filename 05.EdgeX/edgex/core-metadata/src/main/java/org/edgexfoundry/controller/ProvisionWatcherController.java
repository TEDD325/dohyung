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

import org.edgexfoundry.domain.meta.ProvisionWatcher;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;

public interface ProvisionWatcherController {

  /**
   * Fetch a specific Provision Watcher by database generated id. May return null if no
   * ProvisionWatcher matches on id. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Return NotFoundException (HTTP 404) if no ProvisionWatcher is found by
   * the id provided;
   * 
   * @param String ProvisionWatcher database generated id
   * @return ProvisionWatcher
   */
  ProvisionWatcher watcher(@PathVariable String id);

  /**
   * Return all provision watcher objects sorted by database generated id. Returns ServiceException
   * (HTTP 503) for unknown or unanticipated issues. Returns LimitExceededException (HTTP 413) if
   * the number returned exceeds the max limit.
   * 
   * @return list of provision watcher
   */
  List<ProvisionWatcher> watchers();

  /**
   * Return ProvisionWatcher with matching name (name should be unique). May be null if none match.
   * Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Return
   * NotFoundException (HTTP 404) if no ProvisionWatcher is found by the name provided;
   * 
   * @param name
   * @return provision watcher matching on name
   */
  ProvisionWatcher watcherForName(@PathVariable String name);

  /**
   * Find all provision watchers associated to the DeviceProfile with the specified profile database
   * generated identifier. List may be empty if no provision watchers match. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if no DeviceProfile match on the id provided.
   * 
   * @param profile id - device profile's database generated identifier
   * @return List of ProvisionWatchers associated to the device profile
   */
  List<ProvisionWatcher> watchersForProfile(@PathVariable String profileId);

  /**
   * Find all provision watchers associated to the DeviceProfile with the specified profile name.
   * List may be empty if no provision watchers match. Returns ServiceException (HTTP 503) for
   * unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if no DeviceProfile match
   * on the name provided.
   * 
   * @param profile name - provision watcher profile's name
   * @return List of ProvisionWatcher associated to the device profile
   */
  List<ProvisionWatcher> watchersForProfileByName(@PathVariable String profilename);

  /**
   * Find the provision watchers associated to the DeviceService with the specified service database
   * generated identifier. List may be empty if no provision watchers match. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException
   * (HTTP 404) if no DeviceService match on the id provided.
   * 
   * @param service id - device service's database generated identifier
   * @return ProvisionWatcher associated to the device service
   */
  List<ProvisionWatcher> watcherForService(@PathVariable String serviceId);

  /**
   * Find provision watchers associated to the DeviceService with the specified service name. List
   * may be empty if no provision watchers match. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns NotFoundException (HTTP 404) if no DeviceService match on the
   * name provided.
   * 
   * @param service name - provision watcher service's name
   * @return ProvisionWatcher associated to the device service
   */
  List<ProvisionWatcher> watcherForServiceByName(@PathVariable String servicename);

  /**
   * Find ProvisionWatchers by an identifier key/value pair. Returns ServiceException (HTTP 503) for
   * unknown or unanticipated issues.
   * 
   * @param key
   * @param value
   * @return
   */
  List<ProvisionWatcher> watchersForIdentifier(@PathVariable String key,
      @PathVariable String value);

  /**
   * Add a new ProvisionWatcher - name must be unique. Returns ServiceException (HTTP 503) for
   * unknown or unanticipated issues.Returns DataValidationException (HTTP 409) if an associated
   * object (Profile, Service) cannot be found with the id or name provided.
   * 
   * @param ProvisionWatcher to add
   * @return new database generated id for the new ProvisionWatcher
   */
  String add(@RequestBody ProvisionWatcher watcher);

  /**
   * Update the ProvisionWatcher identified by the id or name in the object provided. Id is used
   * first, name is used second for identification purposes. Watcher's service & profile cannot be
   * updated. Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) if no provision watcher with the provided id is found.
   * 
   * @param ProvisionWatcher object holding the identifier and new values for the ProvisionWatcher
   * @return boolean indicating success of the update
   */
  boolean update(@RequestBody ProvisionWatcher watcher2);


  /**
   * Remove the ProvisionWatcher designated by the database generated id for the ProvisoinWatcher.
   * Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) if no provision watcher with the provided id is found.
   * 
   * @param database generated id
   * @return boolean indicating success of the remove operation
   */
  boolean delete(@PathVariable String id);

  /**
   * Remove the ProvisionWatcher designated by unique name identifier. Returns ServiceException
   * (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if no
   * provision watcher with the provided name is found.
   * 
   * @param unique name of the ProvisionWatcher
   * @return boolean indicating success of the remove operation
   */
  boolean deleteByName(@PathVariable String name);



}
