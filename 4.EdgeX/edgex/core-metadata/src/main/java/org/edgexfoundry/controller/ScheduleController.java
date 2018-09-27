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

import org.edgexfoundry.domain.meta.Schedule;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;

public interface ScheduleController {


  /**
   * Fetch a specific Schedule by database generated id. May return null if no schedule with the id
   * is found. Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) when no schedule is found by the id provided.
   * 
   * @param String database generated id for the schedule
   * 
   * @return schedule matching on the id
   */
  Schedule schedule(@PathVariable String id);

  /**
   * Return all schedules sorted by id. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns LimitExceededException (HTTP 413) if the number returned exceeds
   * the max limit.
   * 
   * @return list of schedules
   */
  List<Schedule> schedules();


  /**
   * Return Schedule matching given name (schedule names should be unique). May be null if no
   * schedule matches on the name provided. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns NotFoundException (HTTP 404) when no schedule is found by the
   * name provided.
   * 
   * @param name
   * @return schedule matching on name
   */
  Schedule scheduleForName(@PathVariable String name);

  /**
   * Add a new Schedule - name must be unique. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. DataValidationException (HTTP 409) if any the cron expression string is
   * not properly formatted.
   * 
   * @param Schedule object
   * @return database generated identifier for the new schedule
   */
  String add(@RequestBody Schedule schedule);

  /**
   * Update the Schedule identified by the id or name in the object provided. Id is used first, name
   * is used second for identification purposes. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. DataValidationException (HTTP // * 409) if any the cron expression string
   * is not properly formatted. NotFoundException (HTTP 404) if no schedule is found for the id.
   * 
   * @param schedule2 - object holding the identifier and new values for the schedule
   * @return boolean indicating success of the update
   */
  boolean update(@RequestBody Schedule schedule2);


  /**
   * Remove the Schedule designated by database generated id. ServiceException (HTTP 503) for
   * unknown or unanticipated issues. NotFoundException (HTTP 404) if no Schedule is found with the
   * provided id.
   * 
   * @param database generated id for the Schedule
   * @return boolean indicating success of the remove operation
   */
  boolean delete(@PathVariable String id);

  /**
   * Remove the Schedule designated by name. ServiceException (HTTP 503) for unknown or
   * unanticipated issues. NotFoundException (HTTP 404) if no Schedule is found with the provided
   * name.
   * 
   * @param name for the Schedule
   * @return boolean indicating success of the remove operation
   */
  boolean deleteByName(@PathVariable String name);


}
