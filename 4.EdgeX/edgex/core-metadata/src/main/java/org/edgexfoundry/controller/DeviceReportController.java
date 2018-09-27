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

import org.edgexfoundry.domain.meta.DeviceReport;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;

public interface DeviceReportController {

  /**
   * Fetch a specific DeviceReport by database generated id. May return null if no report with the
   * id is found. Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) if the report cannot be found by id;
   * 
   * @param String database generated id for the report
   * 
   * @return device report matching on the id
   */
  DeviceReport deviceReport(@PathVariable String id);

  /**
   * Return all device reports sorted by id. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns LimitExceededException (HTTP 413) if the number returned exceeds
   * the max limit.
   * 
   * @return list of device reports
   */
  List<DeviceReport> deviceReports();

  /**
   * Return DeviceReport matching given name (device report names should be unique). May be null if
   * no report matches on the name provided. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues.
   * 
   * @param name
   * @return device report matching on name
   */
  DeviceReport deviceReportForName(@PathVariable String name);

  /**
   * Return a list of value descriptor names - this list is the union of all value descriptor names
   * from all the device reports associated to the named device. Returns ServiceException (HTTP 503)
   * for unknown or unanticipated issues.
   * 
   * @param devicename - the unique name of the device that has device reports
   * @return - list of value descriptor unique names
   */
  List<String> associatedValueDescriptors(@PathVariable String devicename);

  List<DeviceReport> deviceReportsForDevice(@PathVariable String devicename);

  /**
   * Add a new DeviceReport - name must be unique. Referenced objects (device, schedule event) are
   * all referenced in the new DeviceReport by name and must already be persisted. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. NotFoundException (HTTP 404)
   * if any referenced object cannot be found by its provided name.
   * 
   * @param DeviceReport object
   * @return database generated identifier for the new device report
   */
  String add(@RequestBody DeviceReport deviceReport);

  /**
   * Update the DeviceReport identified by the id or name in the object provided. Id is used first,
   * name is used second for identification purposes. Returns ServiceException (HTTP 503) for
   * unknown or unanticipated issues. NotFoundException (HTTP 404) if any referenced object cannot
   * be found by its provided name.
   * 
   * @param deviceReport2 - object holding the identifier and new values for the DeviceReport
   * @return boolean indicating success of the update
   */
  boolean update(@RequestBody DeviceReport deviceReport2);

  /**
   * Remove the DevicReport designated by database generated id. ServiceException (HTTP 503) for
   * unknown or unanticipated issues. NotFoundException (HTTP 404) if no DeviceReport is found with
   * the provided id.
   * 
   * @param database generated id for the DeviceReport
   * @return boolean indicating success of the remove operation
   */
  boolean delete(@PathVariable String id);

  /**
   * Remove the DevicReport designated by name. ServiceException (HTTP 503) for unknown or
   * unanticipated issues. NotFoundException (HTTP 404) if no DeviceReport is found with the
   * provided name.
   * 
   * @param name for the DeviceReport
   * @return boolean indicating success of the remove operation
   */
  boolean deleteByName(@PathVariable String name);

}
