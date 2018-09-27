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

package org.edgexfoundry.controller.impl;

import java.util.List;

import org.edgexfoundry.controller.CommandController;
import org.edgexfoundry.dao.CommandRepository;
import org.edgexfoundry.dao.DeviceProfileDao;
import org.edgexfoundry.domain.common.ValueDescriptor;
import org.edgexfoundry.domain.meta.Command;
import org.edgexfoundry.domain.meta.DeviceProfile;
import org.edgexfoundry.exception.controller.DataValidationException;
import org.edgexfoundry.exception.controller.LimitExceededException;
import org.edgexfoundry.exception.controller.NotFoundException;
import org.edgexfoundry.exception.controller.ServiceException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.domain.Sort;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/command")
public class CommandControllerImpl implements CommandController {

  private static final org.edgexfoundry.support.logging.client.EdgeXLogger logger =
      org.edgexfoundry.support.logging.client.EdgeXLoggerFactory
          .getEdgeXLogger(CommandControllerImpl.class);

  @Autowired
  private CommandRepository repos;

  @Autowired
  private DeviceProfileDao profileDao;

  @Value("${read.max.limit}")
  private int maxLimit;

  /**
   * Fetch a specific command by database generated id. May return null if no commands with the id
   * is found. Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) if not found by id.
   * 
   * @param String command id (ObjectId)
   * 
   * @return Command
   */
  @RequestMapping(value = "/{id}", method = RequestMethod.GET)
  @Override
  public Command command(@PathVariable String id) {
    try {
      Command cmd = repos.findOne(id);
      if (cmd == null)
        throw new NotFoundException(Command.class.toString(), id);
      return cmd;
    } catch (NotFoundException nfE) {
      throw nfE;
    } catch (Exception e) {
      logger.error("Error getting command:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Return all command objects. Returns ServiceException (HTTP 503) for unknown or unanticipated
   * issues. Returns LimitExceededException (HTTP 413) if the number returned exceeds the max limit.
   * 
   * @return list of command
   */
  @RequestMapping(method = RequestMethod.GET)
  @Override
  public List<Command> commands() {
    try {
      if (repos.count() > maxLimit) {
        logger.error("Max limit exceeded in request for commands");
        throw new LimitExceededException("Command");
      }
      Sort sort = new Sort(Sort.Direction.DESC, "_id");
      return repos.findAll(sort);
    } catch (LimitExceededException lE) {
      throw lE;
    } catch (Exception e) {
      logger.error("Error getting commands:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Return Command object with given name. Name is not unique for all of EdgeX but is unique per
   * any associated Device Profile. Returns ServiceException (HTTP 503) for unknown or unanticipated
   * issues.
   * 
   * @param name
   * @return list of Commands with matching name
   */
  @RequestMapping(value = "/name/{name:.+}", method = RequestMethod.GET)
  @Override
  public List<Command> commandForName(@PathVariable String name) {
    try {
      return repos.findByName(name);
    } catch (Exception e) {
      logger.error("Error getting command:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Add a new Command. Returns ServiceException (HTTP 503) for unknown or unanticipated issues.
   * 
   * @param Command object
   * @return database generated id for the new command
   */
  @RequestMapping(method = RequestMethod.POST)
  @Override
  public String add(@RequestBody Command command) {
    if (command == null)
      throw new ServiceException(new DataValidationException("No command data provided"));
    try {
      repos.save(command);
      return command.getId();
    } catch (Exception e) {
      logger.error("Error adding Command:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Update the Command identified by the database generated id in the object provided. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. NotFoundException (HTTP 404)
   * if no Command is found with the provided id. DataValidationException (HTTP 409) if the name is
   * updated and it is not unique for the Device Profile.
   * 
   * @param object holding the database generated identifier and new values for the Command
   * @return boolean indicating success of the update
   */
  @RequestMapping(method = RequestMethod.PUT)
  @Override
  public boolean update(@RequestBody Command command2) {
    if (command2 == null)
      throw new ServiceException(new DataValidationException("No command data provided"));
    try {
      Command command = repos.findOne(command2.getId());
      if (command == null) {
        logger.error("Request to update with non-existent or unidentified command (id/name):  "
            + command2.getId() + "/" + command2.getName());
        throw new NotFoundException(ValueDescriptor.class.toString(), command2.getId());
      }
      updateCommand(command2, command);
      return true;
    } catch (NotFoundException nE) {
      throw nE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error updating command:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  private void updateCommand(Command from, Command to) {
    if (from.getGet() != null)
      to.setGet(from.getGet());
    if (from.getPut() != null)
      to.setPut(from.getPut());
    if (from.getName() != null && !from.getName().equals(to.getName())) {
      checkAssociatedProfilesForDupeNames(from);
      to.setName(from.getName());
    }
    if (from.getOrigin() != 0)
      to.setOrigin(from.getOrigin());
    repos.save(to);
  }

  private void checkAssociatedProfilesForDupeNames(Command command) {
    // check any associated profiles to insure new names are not duplicate
    // for that profile. Command names are not globally unique but must be
    // unique per DeviceProfile.
    List<DeviceProfile> profiles = profileDao.getAssociatedProfilesForCommand(command);
    profiles.forEach(p -> profileDao.checkCommandNames(p.getCommands(), command.getName()));
  }

  /**
   * Remove the Command designated by database generated id. ServiceException (HTTP 503) for unknown
   * or unanticipated issues. NotFoundException (HTTP 404) if no Command is found with the provided
   * id. DataValidationException (HTTP 409) if the Command is still associated to a Device Profile.
   * 
   * @param database generated id for the Command
   * @return boolean indicating success of the remove operation
   */
  @RequestMapping(value = "/id/{id}", method = RequestMethod.DELETE)
  @Override
  public boolean delete(@PathVariable String id) {
    try {
      Command command = repos.findOne(id);
      if (command == null) {
        logger.error("Request to delete with non-existent command id:  " + id);
        throw new NotFoundException(Command.class.toString(), id);
      }
      if (profileDao.getAssociatedProfilesForCommand(command).isEmpty()) {
        repos.delete(command);
        return true;
      } else {
        logger.error("Command is still referenced by DeviceProfiles - cannot be deleted");
        throw new DataValidationException(
            "Command is still referenced by DeviceProfiles - cannot be deleted");
      }
    } catch (NotFoundException nE) {
      throw nE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error removing command:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }
}
