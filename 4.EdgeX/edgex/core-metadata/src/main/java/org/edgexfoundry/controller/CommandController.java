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

import org.edgexfoundry.domain.meta.Command;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;

public interface CommandController {

  /**
   * Fetch a specific command by database generated id. May return null if no commands with the id
   * is found. Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) if not found by id.
   * 
   * @param String command id (ObjectId)
   * 
   * @return Command
   */
  Command command(@PathVariable String id);

  /**
   * Return all command objects. Returns ServiceException (HTTP 503) for unknown or unanticipated
   * issues. Returns LimitExceededException (HTTP 413) if the number returned exceeds the max limit.
   * 
   * @return list of command
   */
  List<Command> commands();

  /**
   * Return Command object with given name. Name is not unique for all of EdgeX but is unique per
   * any associated Device Profile. Returns ServiceException (HTTP 503) for unknown or unanticipated
   * issues.
   * 
   * @param name
   * @return list of Commands with matching name
   */
  List<Command> commandForName(@PathVariable String name);

  /**
   * Add a new Command. Returns ServiceException (HTTP 503) for unknown or unanticipated issues.
   * 
   * @param Command object
   * @return database generated id for the new command
   */
  String add(@RequestBody Command command);

  /**
   * Update the Command identified by the database generated id in the object provided. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. NotFoundException (HTTP 404)
   * if no Command is found with the provided id. DataValidationException (HTTP 409) if the name is
   * updated and it is not unique for the Device Profile.
   * 
   * @param object holding the database generated identifier and new values for the Command
   * @return boolean indicating success of the update
   */
  boolean update(@RequestBody Command command2);

  /**
   * Remove the Command designated by database generated id. ServiceException (HTTP 503) for unknown
   * or unanticipated issues. NotFoundException (HTTP 404) if no Command is found with the provided
   * id. DataValidationException (HTTP 409) if the Command is still associated to a Device Profile.
   * 
   * @param database generated id for the Command
   * @return boolean indicating success of the remove operation
   */
  boolean delete(@PathVariable String id);
}
