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

import org.edgexfoundry.controller.Action;
import org.edgexfoundry.controller.ScheduleEventController;
import org.edgexfoundry.dao.AddressableDao;
import org.edgexfoundry.dao.ScheduleEventDao;
import org.edgexfoundry.dao.ScheduleEventRepository;
import org.edgexfoundry.dao.ScheduleRepository;
import org.edgexfoundry.domain.meta.ActionType;
import org.edgexfoundry.domain.meta.Addressable;
import org.edgexfoundry.domain.meta.Schedule;
import org.edgexfoundry.domain.meta.ScheduleEvent;
import org.edgexfoundry.exception.controller.DataValidationException;
import org.edgexfoundry.exception.controller.LimitExceededException;
import org.edgexfoundry.exception.controller.NotFoundException;
import org.edgexfoundry.exception.controller.ServiceException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.data.domain.Sort;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/scheduleevent")
public class ScheduleEventControllerImpl implements ScheduleEventController {

  private static final org.edgexfoundry.support.logging.client.EdgeXLogger logger =
      org.edgexfoundry.support.logging.client.EdgeXLoggerFactory
          .getEdgeXLogger(ScheduleEventControllerImpl.class);

  private static final String ERR_REFBY = "is still referenced by existing DeviceReport.";

  private static final String ERR_GET = "Error getting Schedule Event:  ";

  private static final String ERR_NOT_FND = " was not found.";

  @Autowired
  private ScheduleEventRepository repos;

  @Autowired
  private ScheduleRepository scheduleRepos;

  @Autowired
  private ScheduleEventDao dao;

  @Autowired
  private AddressableDao addressableDao;

  @Autowired
  private CallbackExecutor callback;

  @Value("${read.max.limit}")
  private int maxLimit;

  /**
   * Fetch a specific ScheduleEvent by database generated id. May return null if no schedule event
   * with the id is found. Returns ServiceException (HTTP 503) for unknown or unanticipated issues.
   * Returns NotFoundException (HTTP 404) if a ScheduleEvent cannot be found by the id.
   * 
   * @param String database generated id for the schedule event
   * 
   * @return schedule event matching on the id
   */
  @RequestMapping(value = "/{id}", method = RequestMethod.GET)
  @Override
  public ScheduleEvent scheduleEvent(@PathVariable String id) {
    try {
      ScheduleEvent se = repos.findOne(id);
      if (se == null)
        throw new NotFoundException(ScheduleEvent.class.toString(), id);
      return se;
    } catch (NotFoundException nfE) {
      throw nfE;
    } catch (Exception e) {
      logger.error("Error getting ScheduleEvent:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Return all schedule events sorted by id. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns LimitExceededException (HTTP 413) if the number returned exceeds
   * the max limit.
   * 
   * @return list of schedule events
   */
  @RequestMapping(method = RequestMethod.GET)
  @Override
  public List<ScheduleEvent> scheduleEvents() {
    try {
      if (repos.count() > maxLimit) {
        logger.error("Max limit exceeded in request for schedule events");
        throw new LimitExceededException("ScheduleEvent");
      } else {
        Sort sort = new Sort(Sort.Direction.DESC, "_id");
        return repos.findAll(sort);
      }
    } catch (LimitExceededException lE) {
      throw lE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error getting ScheduleEvents:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Return ScheduleEvents matching given name (schedule names should be unique). May be null if no
   * schedule events matches on the name provided. Returns ServiceException (HTTP 503) for unknown
   * or unanticipated issues.Returns NotFoundException (HTTP 404) if a ScheduleEvent cannot be found
   * by the name.
   * 
   * @param name
   * @return schedule event matching on name
   */
  @RequestMapping(value = "/name/{name:.+}", method = RequestMethod.GET)
  @Override
  public ScheduleEvent scheduleEventForName(@PathVariable String name) {
    try {
      ScheduleEvent se = repos.findByName(name);
      if (se == null)
        throw new NotFoundException(ScheduleEvent.class.toString(), name);
      return se;
    } catch (NotFoundException nfE) {
      throw nfE;
    } catch (Exception e) {
      logger.error("Error getting ScheduleEvent:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Find all schedule events associated to the Addressable with the specified addressable database
   * generated identifier. List may be empty if no schedule events match. Returns ServiceException
   * (HTTP 503) for unknown or unanticipated issues. Returns NotFoundException (HTTP 404) if no
   * Addressable match on the id provided.
   * 
   * @param addressable id - addressable's database generated identifier
   * @return List of ScheduleEvents associated to the addressable
   */
  @RequestMapping(method = RequestMethod.GET, value = "/addressable/{addressableId}")
  @Override
  public List<ScheduleEvent> scheduleEventsForAddressable(@PathVariable String addressableId) {
    try {
      Addressable addressable = addressableDao.getById(addressableId);
      if (addressable == null) {
        logger.error("Request for Schedule Events by non-existent addressable:  " + addressableId);
        throw new NotFoundException(Addressable.class.toString(), addressableId);
      }
      return repos.findByAddressable(addressable);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Find all schedule events associated to the Addressable with the specified addressable name.
   * List may be empty if no schedule events match. Returns ServiceException (HTTP 503) for unknown
   * or unanticipated issues. Returns NotFoundException (HTTP 404) if no Addressable match on the
   * name provided.
   * 
   * @param addressable name - addressable's name
   * @return List of Schedule Events associated to the addressable
   */
  @RequestMapping(method = RequestMethod.GET, value = "/addressablename/{addressablename:.+}")
  @Override
  public List<ScheduleEvent> scheduleEventsForAddressableByName(
      @PathVariable String addressablename) {
    try {
      Addressable addressable = addressableDao.getByName(addressablename);
      if (addressable == null) {
        logger
            .error("Request for Schedule Events by non-existent addressable:  " + addressablename);
        throw new NotFoundException(Addressable.class.toString(), addressablename);
      }
      return repos.findByAddressable(addressable);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Find all schedule events associated to the service with the specified service name. List may be
   * empty if no service names match. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns NotFoundException (HTTP 404) if no Schedule Events match on the
   * name provided.
   * 
   * @param addressable name - addressable's name
   * @return List of Schedule Events associated to the addressable
   */
  @RequestMapping(method = RequestMethod.GET, value = "/servicename/{servicename:.+}")
  @Override
  public List<ScheduleEvent> scheduleEventsForServiceByName(@PathVariable String servicename) {
    try {
      if (isServiceNameValid(servicename)) {
        return repos.findByService(servicename);
      } else {
        logger.error("Service with name: " + servicename + ERR_NOT_FND);
        throw new NotFoundException("Service", servicename);
      }
    } catch (NotFoundException nE) {
      throw nE;
    } catch (Exception e) {
      logger.error(ERR_GET + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Add a new ScheduleEvent - name must be unique. Returns ServiceException (HTTP 503) for unknown
   * or unanticipated issues. NotFoundException (HTTP 404) if the event's associated schedule is not
   * found (referenced by name). DataValidationException (HTTP 409) if the schedule was not
   * provided.
   * 
   * @param Schedule object
   * @return database generated identifier for the new schedule
   */
  @RequestMapping(method = RequestMethod.POST)
  @Override
  public String add(@RequestBody ScheduleEvent scheduleEvent) {
    if (scheduleEvent == null)
      throw new DataValidationException("No schedule event data provided");
    if (scheduleEvent.getSchedule() == null)
      throw new DataValidationException("No schedule provided for schedule event");
    try {
      if (isScheduleNameValid(scheduleEvent.getSchedule())) {
        attachAssociated(scheduleEvent);
        repos.save(scheduleEvent);
        notifyAssociates(scheduleEvent, Action.POST);
        return scheduleEvent.getId();
      } else {
        logger.error("Schedule with name: " + scheduleEvent.getSchedule() + ERR_NOT_FND);
        throw new NotFoundException(Schedule.class.toString(), scheduleEvent.getSchedule());
      }
    } catch (DuplicateKeyException dE) {
      throw new DataValidationException("Name is not unique: " + scheduleEvent.getName());
    } catch (NotFoundException nE) {
      throw nE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error adding ScheduleEvent:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  private void attachAssociated(ScheduleEvent event) {
    Addressable addressable = addressableDao.getByIdOrName(event.getAddressable());
    if (addressable == null)
      throw new DataValidationException(
          "A schedule event must be associated to a known addressable.");
    event.setAddressable(addressable);
  }

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
  @RequestMapping(method = RequestMethod.PUT)
  @Override
  public boolean update(@RequestBody ScheduleEvent scheduleEvent2) {
    if (scheduleEvent2 == null)
      throw new ServiceException(new DataValidationException("No schedule event data provided"));
    try {
      ScheduleEvent scheduleEvent = dao.getByIdOrName(scheduleEvent2);
      if (scheduleEvent == null) {
        logger
            .error("Request to update with non-existent or unidentified ScheduleEvent (id/name):  "
                + scheduleEvent2.getId() + "/" + scheduleEvent2.getName());
        throw new NotFoundException(ScheduleEvent.class.toString(), scheduleEvent2.getId());
      }
      updateScheduleEvent(scheduleEvent2, scheduleEvent);
      return true;
    } catch (NotFoundException nE) {
      throw nE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error updating ScheduleEvent:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  private void updateScheduleEvent(ScheduleEvent from, ScheduleEvent to) {
    boolean serviceChanged = false;
    if (from.getAddressable() != null)
      to.setAddressable(addressableDao.getByIdOrName(from.getAddressable()));
    if (from.getService() != null)
      if (isServiceNameValid(from.getService())) {
        serviceChanged = to.getService() != from.getService();
        to.setService(from.getService());
      } else {
        logger.error("Service with name: " + from.getService() + ERR_NOT_FND);
        throw new NotFoundException("Schedule", from.getService());
      }
    checkSchedule(from, to);
    checkServiceName(from, to);
    if (from.getOrigin() != 0)
      to.setOrigin(from.getOrigin());
    repos.save(to);
    if (serviceChanged) {
      // remove from the from
      notifyAssociates(from, Action.DELETE);
      // add to the to
      notifyAssociates(to, Action.POST);
    } else {
      // update the to
      notifyAssociates(to, Action.PUT);
    }
  }

  private void checkSchedule(ScheduleEvent from, ScheduleEvent to) {
    if (from.getSchedule() != null)
      if (isScheduleNameValid(from.getSchedule()))
        to.setSchedule(from.getSchedule());
      else {
        logger.error("Schedule with name: " + from.getSchedule() + ERR_NOT_FND);
        throw new NotFoundException(Schedule.class.toString(), from.getSchedule());
      }
  }

  private void checkServiceName(ScheduleEvent from, ScheduleEvent to) {
    if (from.getName() != null) {
      if (dao.isScheduleEventAssociatedToDeviceReport(to)) {
        logger
            .error("Data integrity issue. ScheduleEvent with name: " + from.getName() + ERR_REFBY);
        throw new DataValidationException(
            "Data integrity issue. ScheduleEvent with name: " + from.getName() + ERR_REFBY);
      } else
        to.setName(from.getName());
    }
  }

  /**
   * Remove the ScheduleEvent designated by database generated id. ServiceException (HTTP 503) for
   * unknown or unanticipated issues. NotFoundException (HTTP 404) if no ScheduleEvent is found with
   * the provided id. DataValidationException (HTTP 409) if an attempt to delete a schedule event
   * still being referenced by device reports.
   * 
   * @param database generated id for the ScheduleEvent
   * @return boolean indicating success of the remove operation
   */
  @RequestMapping(value = "/id/{id}", method = RequestMethod.DELETE)
  @Override
  public boolean delete(@PathVariable String id) {
    try {
      ScheduleEvent scheduleEvent = repos.findOne(id);
      if (scheduleEvent == null) {
        logger.error("Request to delete with non-existent ScheduleEvent id:  " + id);
        throw new NotFoundException(ScheduleEvent.class.toString(), id);
      }
      notifyAssociates(scheduleEvent, Action.DELETE);
      return deleteScheduleEvent(scheduleEvent);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error removing ScheduleEvent:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Remove the ScheduleEvent designated by name. ServiceException (HTTP 503) for unknown or
   * unanticipated issues. NotFoundException (HTTP 404) if no ScheduleEvent is found with the
   * provided name. DataValidationException (HTTP 409) if an attempt to delete a schedule event
   * still being referenced by device reports.
   * 
   * @param name for the schedule event
   * @return boolean indicating success of the remove operation
   */
  @RequestMapping(value = "/name/{name:.+}", method = RequestMethod.DELETE)
  @Override
  public boolean deleteByName(@PathVariable String name) {
    try {
      ScheduleEvent scheduleEvent = repos.findByName(name);
      if (scheduleEvent == null) {
        logger.error("Request to delete with unknown ScheduleEvent name:  " + name);
        throw new NotFoundException(ScheduleEvent.class.toString(), name);
      }
      notifyAssociates(scheduleEvent, Action.DELETE);
      return deleteScheduleEvent(scheduleEvent);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error removing ScheduleEvent:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  private boolean deleteScheduleEvent(ScheduleEvent scheduleEvent) {
    if (dao.isScheduleEventAssociatedToDeviceReport(scheduleEvent)) {
      logger.error(
          "Data integrity issue. ScheduleEvent with id: " + scheduleEvent.getId() + ERR_REFBY);
      throw new DataValidationException(
          "Data integrity issue. ScheduleEvent with id: " + scheduleEvent.getId() + ERR_REFBY);
    }
    repos.delete(scheduleEvent);
    return true;
  }

  private boolean isScheduleNameValid(String name) {
    return scheduleRepos.findByName(name) != null;
  }

  private boolean isServiceNameValid(String name) {
    // TODO: Fixup once there is a service registry and Service base class
    return true;
  }

  private void notifyAssociates(ScheduleEvent event, Action action) {
    callback.callback(dao.getAffectedService(event), event.getId(), action,
        ActionType.SCHEDULEEVENT);
  }

}
