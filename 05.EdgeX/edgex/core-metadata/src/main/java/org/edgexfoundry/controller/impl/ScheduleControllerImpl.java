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
import org.edgexfoundry.controller.ScheduleController;
import org.edgexfoundry.dao.ScheduleDao;
import org.edgexfoundry.dao.ScheduleRepository;
import org.edgexfoundry.domain.meta.ActionType;
import org.edgexfoundry.domain.meta.Schedule;
import org.edgexfoundry.exception.controller.DataValidationException;
import org.edgexfoundry.exception.controller.LimitExceededException;
import org.edgexfoundry.exception.controller.NotFoundException;
import org.edgexfoundry.exception.controller.ServiceException;
import org.quartz.CronExpression;
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
@RequestMapping("/api/v1/schedule")
public class ScheduleControllerImpl implements ScheduleController {

  private static final org.edgexfoundry.support.logging.client.EdgeXLogger logger =
      org.edgexfoundry.support.logging.client.EdgeXLoggerFactory
          .getEdgeXLogger(ScheduleControllerImpl.class);

  @Autowired
  private ScheduleRepository repos;

  @Autowired
  private ScheduleDao dao;

  @Autowired
  private CallbackExecutor callback;

  @Value("${read.max.limit}")
  private int maxLimit;

  /**
   * Fetch a specific Schedule by database generated id. May return null if no schedule with the id
   * is found. Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) when no schedule is found by the id provided.
   * 
   * @param String database generated id for the schedule
   * 
   * @return schedule matching on the id
   */
  @RequestMapping(value = "/{id}", method = RequestMethod.GET)
  @Override
  public Schedule schedule(@PathVariable String id) {
    try {
      Schedule schedule = repos.findOne(id);
      if (schedule == null)
        throw new NotFoundException(Schedule.class.toString(), id);
      return schedule;
    } catch (NotFoundException nfE) {
      throw nfE;
    } catch (Exception e) {
      logger.error("Error getting schedule:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Return all schedules sorted by id. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns LimitExceededException (HTTP 413) if the number returned exceeds
   * the max limit.
   * 
   * @return list of schedules
   */
  @RequestMapping(method = RequestMethod.GET)
  @Override
  public List<Schedule> schedules() {
    try {
      if (repos.count() > maxLimit) {
        logger.error("Max limit exceeded in request for schedules");
        throw new LimitExceededException("Schedule");
      } else {
        Sort sort = new Sort(Sort.Direction.DESC, "_id");
        return repos.findAll(sort);
      }
    } catch (LimitExceededException lE) {
      throw lE;
    } catch (Exception e) {
      logger.error("Error getting schedules:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Return Schedule matching given name (schedule names should be unique). May be null if no
   * schedule matches on the name provided. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns NotFoundException (HTTP 404) when no schedule is found by the
   * name provided.
   * 
   * @param name
   * @return schedule matching on name
   */
  @RequestMapping(value = "/name/{name:.+}", method = RequestMethod.GET)
  @Override
  public Schedule scheduleForName(@PathVariable String name) {
    try {
      Schedule schedule = repos.findByName(name);
      if (schedule == null)
        throw new NotFoundException(Schedule.class.toString(), name);
      return schedule;
    } catch (NotFoundException nfE) {
      throw nfE;
    } catch (Exception e) {
      logger.error("Error getting Schedule:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Add a new Schedule - name must be unique. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. DataValidationException (HTTP 409) if any the cron expression string is
   * not properly formatted.
   * 
   * @param Schedule object
   * @return database generated identifier for the new schedule
   */
  @RequestMapping(method = RequestMethod.POST)
  @Override
  public String add(@RequestBody Schedule schedule) {
    if (schedule == null)
      throw new ServiceException(new DataValidationException("No schedule data provided"));
    try {
      if (schedule.getCron() != null) {
        validateCronExpression(schedule.getCron());
      }
      repos.save(schedule);
      notifyAssociates(schedule, Action.POST);
      return schedule.getId();
    } catch (DuplicateKeyException dE) {
      throw new DataValidationException("Name is not unique: " + schedule.getName());
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error adding schedule:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Update the Schedule identified by the id or name in the object provided. Id is used first, name
   * is used second for identification purposes. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. DataValidationException (HTTP // * 409) if any the cron expression string
   * is not properly formatted. NotFoundException (HTTP 404) if no schedule is found for the id.
   * 
   * @param schedule2 - object holding the identifier and new values for the schedule
   * @return boolean indicating success of the update
   */
  @RequestMapping(method = RequestMethod.PUT)
  @Override
  public boolean update(@RequestBody Schedule schedule2) {
    if (schedule2 == null)
      throw new ServiceException(new DataValidationException("No schedule data provided"));
    try {
      Schedule schedule = dao.getByIdOrName(schedule2);
      if (schedule == null) {
        logger.error("Request to update with non-existent or unidentified schedule (id/name):  "
            + schedule2.getId() + "/" + schedule2.getName());
        throw new NotFoundException(Schedule.class.toString(), schedule2.getId());
      }
      updateSchedule(schedule2, schedule);
      return true;
    } catch (NotFoundException nE) {
      throw nE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error updating Schedule:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  private void updateSchedule(Schedule from, Schedule to) {
    if (from.getCron() != null) {
      validateCronExpression(from.getCron());
      to.setCron(from.getCron());
    }
    if (from.getEnd() != null)
      to.setEnd(from.getEnd());
    if (from.getFrequency() != null)
      to.setFrequency(from.getFrequency());
    if (from.getStart() != null)
      to.setStart(from.getStart());
    if (from.getName() != null && !from.getName().equals(to.getName()))
      if (!dao.isScheduleAssociatedToScheduleEvent(to))
        to.setName(from.getName());
      else
        throw new DataValidationException(
            "Schedule's name cannot be changed while associated to an existing ScheduleEvent");
    if (from.getOrigin() != 0)
      to.setOrigin(from.getOrigin());
    repos.save(to);
    notifyAssociates(to, Action.PUT);
  }

  /**
   * Remove the Schedule designated by database generated id. ServiceException (HTTP 503) for
   * unknown or unanticipated issues. NotFoundException (HTTP 404) if no Schedule is found with the
   * provided id.
   * 
   * @param database generated id for the Schedule
   * @return boolean indicating success of the remove operation
   */
  @RequestMapping(value = "/id/{id}", method = RequestMethod.DELETE)
  @Override
  public boolean delete(@PathVariable String id) {
    try {
      Schedule schedule = repos.findOne(id);
      if (schedule == null) {
        logger.error("Request to delete with non-existent Schedule id:  " + id);
        throw new NotFoundException(Schedule.class.toString(), id);
      }
      notifyAssociates(schedule, Action.DELETE);
      return deleteSchedule(schedule);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error removing Schedule:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Remove the Schedule designated by name. ServiceException (HTTP 503) for unknown or
   * unanticipated issues. NotFoundException (HTTP 404) if no Schedule is found with the provided
   * name.
   * 
   * @param name for the Schedule
   * @return boolean indicating success of the remove operation
   */
  @RequestMapping(value = "/name/{name:.+}", method = RequestMethod.DELETE)
  @Override
  public boolean deleteByName(@PathVariable String name) {
    try {
      Schedule schedule = repos.findByName(name);
      if (schedule == null) {
        logger.error("Request to delete with unknown schedule name:  " + name);
        throw new NotFoundException(Schedule.class.toString(), name);
      }
      notifyAssociates(schedule, Action.DELETE);
      return deleteSchedule(schedule);
    } catch (NotFoundException nE) {
      throw nE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error removing schedule:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  private boolean deleteSchedule(Schedule schedule) {
    if (dao.isScheduleAssociatedToScheduleEvent(schedule)) {
      logger.error("Data integrity issue. Schedule with id: " + schedule.getId()
          + "is still referenced by existing ScheduleEvents.");
      throw new DataValidationException("Data integrity issue. Schedule with id: "
          + schedule.getId() + "is still referenced by existing ScheduleEvents.");
    }
    repos.delete(schedule);
    return true;
  }

  private void validateCronExpression(String expression) {
    if (!CronExpression.isValidExpression(expression)) {
      logger.error("Data integrity issue. Schedule's cron expression is invalid:  " + expression);
      throw new DataValidationException(
          "Data integrity issue. Schedule's cron expression is invalid:  " + expression);
    }
  }

  private void notifyAssociates(Schedule schedule, Action action) {
    callback.callback(dao.getAffectedServices(schedule), schedule.getId(), action,
        ActionType.SCHEDULE);
  }
}
