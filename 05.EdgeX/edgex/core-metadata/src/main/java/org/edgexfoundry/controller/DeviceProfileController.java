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

import org.edgexfoundry.domain.meta.DeviceProfile;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

public interface DeviceProfileController {

  /**
   * Fetch a specific profile by database generated id. May return null if no profile with the id is
   * found. Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) if no profile can not be found by id.
   * 
   * @param String database generated id for the profile
   * 
   * @return device profile matching on id
   */
  DeviceProfile deviceProfile(@PathVariable String id);

  /**
   * Fetch the profile identified by database generated id and return as a YAML string. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues.
   * 
   * @param String database generated id for the profile
   * 
   * @return device profile in YAML format
   */
  String deviceProfileAsYaml(@PathVariable String id);

  /**
   * Return the DeviceProfile matching given name (profile names should be unique). May be null if
   * no profiles matches on the name provided. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns NotFoundException (HTTP 404) if no profile can not be found by
   * name.
   * 
   * @param name
   * @return device profile matching on name
   */
  DeviceProfile deviceProfileForName(@PathVariable String name);

  /**
   * Return, in yaml form, the DeviceProfiles matching given name (profile names should be unique).
   * May be null if no profiles matches on the name provided. Returns ServiceException (HTTP 503)
   * for unknown or unanticipated issues.
   * 
   * @param name
   * @return device profile in YAML matching on name
   */
  String deviceProfileAsYamlForName(@PathVariable String name);

  /**
   * Return all profiles sorted by id. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns LimitExceededException (HTTP 413) if the number returned exceeds
   * the max limit.
   * 
   * @return list of profiles
   */
  List<DeviceProfile> deviceProfiles();

  /**
   * Find all DeviceProfiles with a manufacture attribute matching that provided. List may be empty
   * if no profiles match. Returns ServiceException (HTTP 503) for unknown or unanticipated issues.
   * 
   * @param manufacturer - manufacturer to match
   * @return List of DeviceProfile matching on specified manufacturer.
   */
  List<DeviceProfile> deviceProfilesByManufacturer(@PathVariable String manufacturer);

  /**
   * Find all DeviceProfiles with a model attribute matching that provided. List may be empty if no
   * profiles match. Returns ServiceException (HTTP 503) for unknown or unanticipated issues.
   * 
   * @param model - model to match
   * @return List of DeviceProfile matching on specified model.
   */
  List<DeviceProfile> deviceProfilesByModel(@PathVariable String model);

  /**
   * Find all DeviceProfiles with a manufacture or model attribute matching that provided (either
   * matching provides a hit). List may be empty if no profiles match. Returns ServiceException
   * (HTTP 503) for unknown or unanticipated issues.
   * 
   * @param manufacturer - manufacturer to match
   * @param model - model to match
   * 
   * @return List of DeviceProfile matching on specified manufacturer and/or model.
   */
  List<DeviceProfile> deviceProfilesByManufacturerOrModel(@PathVariable String manufacturer,
      @PathVariable String model);

  /**
   * Find all DeviceProfiles having at least one label matching the label provided. List may be
   * empty if no profiles match. Returns ServiceException (HTTP 503) for unknown or unanticipated
   * issues.
   * 
   * @param label - label to be matched
   * 
   * @return List of DeviceProfile matching on specified label
   */
  List<DeviceProfile> deviceProfilesByLabel(@PathVariable String label);

  /**
   * Add a new DeviceProfile (and associated Command objects) - name must be unique. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * DataValidationException (HTTP 409) if an associated command's name is a duplicate for the
   * profile.
   * 
   * @param DeviceProfile object
   * @return database generated identifier for the new device profile
   */
  String add(@RequestBody DeviceProfile deviceProfile);

  /**
   * Add a new DeviceProfile (and associated Command objects) via YAML profile file - name must be
   * unique. Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * DataValidationException (HTTP 409) if an associated command's name is a duplicate for the
   * profile. Returns ClientException (HTTP 400) if the YAML file is empty.
   * 
   * 
   * @param file - YAML file containing the profile
   * @return database generated identifier for the new device profile
   */
  String uploadYamlFile(@RequestParam("file") MultipartFile file);

  /**
   * Add a new DeviceProfile (and associated Command objects) via YAML content - name must be
   * unique. Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * DataValidationException (HTTP 409) if an associated command's name is a duplicate for the
   * profile.
   * 
   * @param yamlContent - YAML profile content
   * @return database generated identifier for the new device profile
   */
  String uploadYaml(@RequestBody String yamlContent);

  /**
   * Update the DeviceProfile identified by the id or name stored in the object provided. Id is used
   * first, name is used second for identification purposes. Associated commands must be updated
   * directly. Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) if the profile cannot be found by the identifier provided.
   * 
   * @param profile2 - object holding the identifier and new values for the DeviceProfile
   * @return boolean indicating success of the update
   */
  boolean update(@RequestBody DeviceProfile profile2);

  /**
   * Remove the DeviceProfile designated by database generated id. This does not remove associated
   * commands. Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) if the device profile cannot be found by the identifier provided.
   * Returns DataValidationException (HTTP 413) if devices still reference the profile.
   * 
   * @param database generated id for the device profile
   * @return boolean indicating success of the remove operation
   */
  boolean delete(@PathVariable String id);

  /**
   * Remove the DeviceProfile designated by unique name. This does not remove associated commands.
   * Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) if the device profile cannot be found by the name provided.
   * Returns DataValidationException (HTTP 413) if devices still reference the profile.
   * 
   * @param unique name of the device profile
   * @return boolean indicating success of the remove operation
   */
  boolean deleteByName(@PathVariable String name);

}
