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

import org.edgexfoundry.domain.meta.ScheduleEvent;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;

public interface ScheduleEventController {

  /**
   * Fetch a specific ScheduleEvent by database generated id. May return null if no schedule event
   * with the id is found. Returns ServiceException (HTTP 503) for unknown or unanticipated issues.
   * Returns NotFoundException (HTTP 404) if a ScheduleEvent cannot be found by the id.
   * 
   * @param String database generated id for the schedule event
   * 
   * @return schedule event matching on the id
   */
  ScheduleEvent scheduleEvent(@PathVariable String id);

  /**
   * Return all schedule events sorted by id. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns LimitExceededException (HTTP 413) if the number returned exceeds
   * the max limit.
   * 
   * @return list of schedule events
   */
  List<ScheduleEvent> scheduleEvents();

  /**
   * Return ScheduleEvents matching given name (schedule names should be unique). May be null if no
   * schedule events matches on the name provided. Returns ServiceException (HTTP 503) for unknown
   * or unanticipated issues.Returns NotFoundException (HTTP 404) if a ScheduleEvent cannot be found
   * by the name.
   * 
   * @param name
   * @return schedule event matching on name
   */
  ScheduleEvent scheduleEventForName(@PathVariable String name);

  /**
   * Find all schedule events associated to the Addressable with the specified addressable database
   * generated identifier. List may be empty if no schedule events match. Returns ServiceException
   * (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if no
   * Addressable match on the id provided.
   * 
   * @param addressable id - addressable's database generated identifier
   * @return List of ScheduleEvents associated to the addressable
   */
  List<ScheduleEvent> scheduleEventsForAddressable(@PathVariable String addressableId);

  /**
   * Find all schedule events associated to the Addressable with the specified addressable name.
   * List may be empty if no schedule events match. Returns ServiceException (HTTP 503) for unknown
   * or unanticipated issues. Returns NotFoundException (HTTP 404) if no Addressable match on the
   * name provided.
   * 
   * @param addressable name - addressable's name
   * @return List of Schedule Events associated to the addressable
   */
  List<ScheduleEvent> scheduleEventsForAddressableByName(@PathVariable String addressablename);

  /**
   * Find all schedule events associated to the service with the specified service name. List may be
   * empty if no service names match. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns NotFoundException (HTTP 404) if no Schedule Events match on the
   * name provided.
   * 
   * @param addressable name - addressable's name
   * @return List of Schedule Events associated to the addressable
   */
  List<ScheduleEvent> scheduleEventsForServiceByName(@PathVariable String servicename);

  /**
   * Add a new ScheduleEvent - name must be unique. Returns ServiceException (HTTP 503) for unknown
   * or unanticipated issues. NotFoundException (HTTP 404) if the event's associated schedule is not
   * found (referenced by name). DataValidationException (HTTP 409) if the schedule was not
   * provided.
   * 
   * @param Schedule object
   * @return database generated identifier for the new schedule
   */
  String add(@RequestBody ScheduleEvent scheduleEvent);

  /**
   * Update the ScheduleEvent identified by the id or name in the object provided. Id is used first,
   * name is used second for identification purposes. Returns ServiceException (HTTP 503) for
   * unknown or unanticipated issues. DataValidationException (HTTP 409) if an attempt to change the
   * name is made when the schedule event is still being referenced by device reports.
   * NotFoundException (HTTP 404) if no schedule is found for the name provided.
   * 
   * @param schedule2 - object holding the identifier and new values for the schedule event
   * @return boolean indicating success of the update
   */
  boolean update(@RequestBody ScheduleEvent scheduleEvent2);

  /**
   * Remove the ScheduleEvent designated by database generated id. ServiceException (HTTP 503) for
   * unknown or unanticipated issues. NotFoundException (HTTP 404) if no ScheduleEvent is found with
   * the provided id. DataValidationException (HTTP 409) if an attempt to delete a schedule event
   * still being referenced by device reports.
   * 
   * @param database generated id for the ScheduleEvent
   * @return boolean indicating success of the remove operation
   */
  boolean delete(@PathVariable String id);

  /**
   * Remove the ScheduleEvent designated by name. ServiceException (HTTP 503) for unknown or
   * unanticipated issues. NotFoundException (HTTP 404) if no ScheduleEvent is found with the
   * provided name. DataValidationException (HTTP 409) if an attempt to delete a schedule event
   * still being referenced by device reports.
   * 
   * @param name for the schedule event
   * @return boolean indicating success of the remove operation
   */
  boolean deleteByName(@PathVariable String name);


}
