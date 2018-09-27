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
import org.edgexfoundry.controller.DeviceReportController;
import org.edgexfoundry.dao.DeviceReportDao;
import org.edgexfoundry.dao.DeviceReportRepository;
import org.edgexfoundry.dao.DeviceRepository;
import org.edgexfoundry.dao.ScheduleEventRepository;
import org.edgexfoundry.domain.meta.ActionType;
import org.edgexfoundry.domain.meta.Device;
import org.edgexfoundry.domain.meta.DeviceReport;
import org.edgexfoundry.domain.meta.ScheduleEvent;
import org.edgexfoundry.exception.controller.ClientException;
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
@RequestMapping("/api/v1/devicereport")
public class DeviceReportControllerImpl implements DeviceReportController {

  private static final org.edgexfoundry.support.logging.client.EdgeXLogger logger =
      org.edgexfoundry.support.logging.client.EdgeXLoggerFactory
          .getEdgeXLogger(DeviceReportControllerImpl.class);

  @Autowired
  private DeviceReportRepository repos;

  @Autowired
  private DeviceReportDao dao;

  @Autowired
  private ScheduleEventRepository scheduleEventRepos;

  @Autowired
  private DeviceRepository deviceRepos;

  @Value("${read.max.limit}")
  private int maxLimit;

  @Autowired
  private CallbackExecutor callback;

  /**
   * Fetch a specific DeviceReport by database generated id. May return null if no report with the
   * id is found. Returns ServiceException (HTTP 503) for unknown or unanticipated issues. Returns
   * NotFoundException (HTTP 404) if the report cannot be found by id;
   * 
   * @param String database generated id for the report
   * 
   * @return device report matching on the id
   */
  @RequestMapping(value = "/{id}", method = RequestMethod.GET)
  @Override
  public DeviceReport deviceReport(@PathVariable String id) {
    try {
      DeviceReport report = repos.findOne(id);
      if (report == null)
        throw new NotFoundException(DeviceReport.class.toString(), id);
      return report;
    } catch (NotFoundException nfE) {
      throw nfE;
    } catch (Exception e) {
      logger.error("Error getting DeviceReport:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Return all device reports sorted by id. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues. Returns LimitExceededException (HTTP 413) if the number returned exceeds
   * the max limit.
   * 
   * @return list of device reports
   */
  @RequestMapping(method = RequestMethod.GET)
  @Override
  public List<DeviceReport> deviceReports() {
    try {
      if (repos.count() > maxLimit) {
        logger.error("Max limit exceeded in request for devices");
        throw new LimitExceededException("DeviceReport");
      } else {
        Sort sort = new Sort(Sort.Direction.DESC, "_id");
        return repos.findAll(sort);
      }
    } catch (LimitExceededException lE) {
      throw lE;
    } catch (Exception e) {
      logger.error("Error getting DeviceReports:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Return DeviceReport matching given name (device report names should be unique). May be null if
   * no report matches on the name provided. Returns ServiceException (HTTP 503) for unknown or
   * unanticipated issues.
   * 
   * @param name
   * @return device report matching on name
   */
  @RequestMapping(value = "/name/{name:.+}", method = RequestMethod.GET)
  @Override
  public DeviceReport deviceReportForName(@PathVariable String name) {
    try {
      DeviceReport report = repos.findByName(name);
      if (report == null)
        throw new NotFoundException(DeviceReport.class.toString(), name);
      return report;
    } catch (NotFoundException nfE) {
      throw nfE;
    } catch (Exception e) {
      logger.error("Error getting DeviceReport:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Return a list of value descriptor names - this list is the union of all value descriptor names
   * from all the device reports associated to the named device. Returns ServiceException (HTTP 503)
   * for unknown or unanticipated issues.
   * 
   * @param devicename - the unique name of the device that has device reports
   * @return - list of value descriptor unique names
   */
  @RequestMapping(value = "/valueDescriptorsFor/{devicename:.+}", method = RequestMethod.GET)
  @Override
  public List<String> associatedValueDescriptors(@PathVariable String devicename) {
    try {
      return dao.getValueDescriptorsForDeviceReportsAssociatedToDevice(devicename);
    } catch (Exception e) {
      logger.error("Error getting value descriptors associted with device reports for a device:  "
          + e.getMessage());
      throw new ServiceException(e);
    }
  }

  @RequestMapping(value = "/devicename/{devicename:.+}", method = RequestMethod.GET)
  @Override
  public List<DeviceReport> deviceReportsForDevice(@PathVariable String devicename) {
    try {
      return repos.findByDevice(devicename);
    } catch (Exception e) {
      logger.error("Error getting device reports for a device name:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Add a new DeviceReport - name must be unique. Referenced objects (device, schedule event) are
   * all referenced in the new DeviceReport by name and must already be persisted. Returns
   * ServiceException (HTTP 503) for unknown or unanticipated issues. NotFoundException (HTTP 404)
   * if any referenced object cannot be found by its provided name.
   * 
   * @param DeviceReport object
   * @return database generated identifier for the new device report
   */
  @RequestMapping(method = RequestMethod.POST)
  @Override
  public String add(@RequestBody DeviceReport deviceReport) {
    if (deviceReport == null)
      throw new ServiceException(new DataValidationException("No device report data provided"));
    try {
      // TODO - someday check value descriptors exist
      validateDevice(deviceReport.getDevice());
      validateScheduleEvent(deviceReport.getEvent());
      repos.save(deviceReport);
      notifyAssociates(deviceReport, Action.POST);
      return deviceReport.getId();
    } catch (DuplicateKeyException dE) {
      throw new DataValidationException("Name is not unique: " + deviceReport.getName());
    } catch (NotFoundException nE) {
      throw nE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error adding DeviceReport:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Update the DeviceReport identified by the id or name in the object provided. Id is used first,
   * name is used second for identification purposes. Returns ServiceException (HTTP 503) for
   * unknown or unanticipated issues. NotFoundException (HTTP 404) if any referenced object cannot
   * be found by its provided name.
   * 
   * @param deviceReport2 - object holding the identifier and new values for the DeviceReport
   * @return boolean indicating success of the update
   */
  @RequestMapping(method = RequestMethod.PUT)
  @Override
  public boolean update(@RequestBody DeviceReport deviceReport2) {
    if (deviceReport2 == null)
      throw new ServiceException(new DataValidationException("No device report data provided"));
    try {
      DeviceReport deviceReport = dao.getByIdOrName(deviceReport2);
      if (deviceReport == null) {
        logger.error("Request to update with non-existent or unidentified DeviceReport (id/name):  "
            + deviceReport2.getId() + "/" + deviceReport2.getName());
        throw new NotFoundException(DeviceReport.class.toString(), deviceReport2.getId());
      }
      updateDeviceReport(deviceReport2, deviceReport);
      return true;
    } catch (NotFoundException nE) {
      throw nE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (ClientException cE) {
      throw cE;
    } catch (Exception e) {
      logger.error("Error updating DeviceReport:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  private void updateDeviceReport(DeviceReport from, DeviceReport to) {
    if (from.getDevice() != null) {
      validateDevice(from.getDevice());
      to.setDevice(from.getDevice());
    }
    if (from.getEvent() != null) {
      validateScheduleEvent(from.getEvent());
      to.setEvent(from.getEvent());
    }
    if (from.getExpected() != null)
      to.setExpected(from.getExpected());
    // TODO - someday find way to check check value descriptors
    // device report name is not referenced by
    // other objects - therefore no check is
    // necessary
    if (from.getName() != null)
      to.setName(from.getName());
    if (from.getOrigin() != 0)
      to.setOrigin(from.getOrigin());
    repos.save(to);
    notifyAssociates(to, Action.PUT);
  }

  /**
   * Remove the DevicReport designated by database generated id. ServiceException (HTTP 503) for
   * unknown or unanticipated issues. NotFoundException (HTTP 404) if no DeviceReport is found with
   * the provided id.
   * 
   * @param database generated id for the DeviceReport
   * @return boolean indicating success of the remove operation
   */
  @RequestMapping(value = "/id/{id}", method = RequestMethod.DELETE)
  @Override
  public boolean delete(@PathVariable String id) {
    try {
      DeviceReport deviceReport = repos.findOne(id);
      if (deviceReport == null) {
        logger.error("Request to delete with non-existent DeviceReport id:  " + id);
        throw new NotFoundException(DeviceReport.class.toString(), id);
      }
      repos.delete(deviceReport);
      return true;
    } catch (NotFoundException nE) {
      throw nE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error removing DeviceReport:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  /**
   * Remove the DevicReport designated by name. ServiceException (HTTP 503) for unknown or
   * unanticipated issues. NotFoundException (HTTP 404) if no DeviceReport is found with the
   * provided name.
   * 
   * @param name for the DeviceReport
   * @return boolean indicating success of the remove operation
   */
  @RequestMapping(value = "/name/{name:.+}", method = RequestMethod.DELETE)
  @Override
  public boolean deleteByName(@PathVariable String name) {
    try {
      DeviceReport deviceReport = repos.findByName(name);
      if (deviceReport == null) {
        logger.error("Request to delete with unknown DeviceReport name:  " + name);
        throw new NotFoundException(DeviceReport.class.toString(), name);
      }
      repos.delete(deviceReport);
      return true;
    } catch (NotFoundException nE) {
      throw nE;
    } catch (DataValidationException dE) {
      throw dE;
    } catch (Exception e) {
      logger.error("Error removing DeviceReport:  " + e.getMessage());
      throw new ServiceException(e);
    }
  }

  private void validateDevice(String device) {
    if (device == null || deviceRepos.findByName(device) == null) {
      logger.error("DeviceReport references non-existent Device:  " + device);
      throw new NotFoundException(Device.class.toString(), device);
    }
  }

  private void validateScheduleEvent(String scheduleEvent) {
    if (scheduleEvent == null || scheduleEventRepos.findByName(scheduleEvent) == null) {
      logger.error("DeviceReport references non-existent ScheduleEvent:  " + scheduleEvent);
      throw new NotFoundException(ScheduleEvent.class.toString(), scheduleEvent);
    }
  }

  private void notifyAssociates(DeviceReport report, Action action) {
    callback.callback(dao.getOwningService(report), report.getId(), action, ActionType.MANAGER);
  }

}
