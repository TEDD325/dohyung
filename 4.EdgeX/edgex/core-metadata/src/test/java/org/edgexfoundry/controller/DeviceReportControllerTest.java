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
 * @microservice: support-logging
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.controller;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.when;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

import org.edgexfoundry.controller.impl.CallbackExecutor;
import org.edgexfoundry.controller.impl.DeviceReportControllerImpl;
import org.edgexfoundry.dao.DeviceReportDao;
import org.edgexfoundry.dao.DeviceReportRepository;
import org.edgexfoundry.dao.DeviceRepository;
import org.edgexfoundry.dao.ScheduleEventRepository;
import org.edgexfoundry.domain.meta.DeviceReport;
import org.edgexfoundry.exception.controller.DataValidationException;
import org.edgexfoundry.exception.controller.LimitExceededException;
import org.edgexfoundry.exception.controller.NotFoundException;
import org.edgexfoundry.exception.controller.ServiceException;
import org.edgexfoundry.test.category.RequiresNone;
import org.edgexfoundry.test.data.DeviceData;
import org.edgexfoundry.test.data.ReportData;
import org.edgexfoundry.test.data.ScheduleEventData;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.data.domain.Sort;

@Category(RequiresNone.class)
public class DeviceReportControllerTest {

  private static final String LIMIT_PROPERTY = "maxLimit";
  private static final int MAX_LIMIT = 100;

  private static final String TEST_ID = "123";
  private static final String TEST_ERR_MSG = "test message";

  @InjectMocks
  private DeviceReportControllerImpl controller;

  @Mock
  private DeviceReportRepository repos;

  @Mock
  private DeviceReportDao dao;

  @Mock
  private ScheduleEventRepository scheduleEventRepos;

  @Mock
  private DeviceRepository deviceRepos;

  @Mock
  private CallbackExecutor callback;

  private DeviceReport report;

  @Before
  public void setup() throws Exception {
    MockitoAnnotations.initMocks(this);
    setControllerMAXLIMIT(MAX_LIMIT);
    report = ReportData.newTestInstance();
    report.setId(TEST_ID);
  }

  @Test
  public void testDeviceReport() {
    when(repos.findOne(TEST_ID)).thenReturn(report);
    assertEquals("Device Report returned is not as expected", report,
        controller.deviceReport(TEST_ID));
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceReportNotFound() {
    controller.deviceReport(TEST_ID);
  }

  @Test(expected = ServiceException.class)
  public void testDeviceReportException() {
    when(repos.findOne(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.deviceReport(TEST_ID);
  }

  @Test
  public void testDeviceReports() {
    List<DeviceReport> rpts = new ArrayList<>();
    rpts.add(report);
    when(repos.findAll(any(Sort.class))).thenReturn(rpts);
    when(repos.count()).thenReturn(1L);
    List<DeviceReport> reports = controller.deviceReports();
    assertEquals("Number of devices reports returned does not matched expected number", 1,
        reports.size());
    assertEquals("Device returned is not as expected", report, reports.get(0));
  }

  @Test(expected = LimitExceededException.class)
  public void testDeviceReportsMaxLimit() {
    List<DeviceReport> reports = new ArrayList<>();
    reports.add(report);
    when(repos.count()).thenReturn(1000L);
    controller.deviceReports();
  }

  @Test(expected = ServiceException.class)
  public void testDeviceReportsException() {
    List<DeviceReport> rpts = new ArrayList<>();
    rpts.add(report);
    when(repos.findAll(any(Sort.class))).thenThrow(new RuntimeException(TEST_ERR_MSG));
    when(repos.count()).thenReturn(1L);
    controller.deviceReports();
  }

  @Test
  public void testDeviceReportForName() {
    when(repos.findByName(ReportData.TEST_RPT_NAME)).thenReturn(report);
    assertEquals("Device report returned is not as expected", report,
        controller.deviceReportForName(ReportData.TEST_RPT_NAME));
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceReportForNameNotFound() {
    controller.deviceReportForName(ReportData.TEST_RPT_NAME);
  }

  @Test(expected = ServiceException.class)
  public void testDeviceReportForNameException() {
    when(repos.findByName(ReportData.TEST_RPT_NAME)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.deviceReportForName(ReportData.TEST_RPT_NAME);
  }

  @Test
  public void testAssociatedValueDescriptors() {
    List<String> valueDescriptors = new ArrayList<>();
    valueDescriptors.add("Temperature");
    when(dao.getValueDescriptorsForDeviceReportsAssociatedToDevice(ReportData.TEST_RPT_NAME))
        .thenReturn(valueDescriptors);
    assertEquals("Value descriptors returned is not expected", valueDescriptors,
        controller.associatedValueDescriptors(ReportData.TEST_RPT_NAME));
  }

  @Test(expected = ServiceException.class)
  public void testAssociatedValueDescriptorsException() {
    List<String> valueDescriptors = new ArrayList<>();
    valueDescriptors.add("Temperature");
    when(dao.getValueDescriptorsForDeviceReportsAssociatedToDevice(ReportData.TEST_RPT_NAME))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.associatedValueDescriptors(ReportData.TEST_RPT_NAME);
  }

  @Test
  public void testDeviceReportForDevice() {
    List<DeviceReport> rpts = new ArrayList<>();
    rpts.add(report);
    when(repos.findByDevice(DeviceData.TEST_NAME)).thenReturn(rpts);
    List<DeviceReport> reports = controller.deviceReportsForDevice(DeviceData.TEST_NAME);
    assertEquals("Number of devices reports returned does not matched expected number", 1,
        reports.size());
    assertEquals("Device returned is not as expected", report, reports.get(0));
  }

  @Test
  public void testDeviceReportForDeviceNameNotFound() {
    List<DeviceReport> reports = controller.deviceReportsForDevice(DeviceData.TEST_NAME);
    assertEquals("Number of devices reports returned does not matched expected number", 0,
        reports.size());
  }

  @Test(expected = ServiceException.class)
  public void testDeviceReportForDeviceException() {
    List<DeviceReport> rpts = new ArrayList<>();
    rpts.add(report);
    when(repos.findByDevice(DeviceData.TEST_NAME)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.deviceReportsForDevice(DeviceData.TEST_NAME);
  }

  @Test
  public void testAdd() {
    report.setDevice(DeviceData.TEST_NAME);
    report.setEvent(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME);
    when(repos.save(report)).thenReturn(report);
    when(deviceRepos.findByName(DeviceData.TEST_NAME)).thenReturn(DeviceData.newTestInstance());
    when(scheduleEventRepos.findByName(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME))
        .thenReturn(ScheduleEventData.newTestInstance());
    assertEquals("Device Report ID returned is not the value expected", TEST_ID,
        controller.add(report));
  }

  @Test(expected = NotFoundException.class)
  public void testAddNoDevice() {
    report.setDevice(DeviceData.TEST_NAME);
    report.setEvent(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME);
    when(repos.save(report)).thenReturn(report);
    when(deviceRepos.findByName(DeviceData.TEST_NAME)).thenReturn(null);
    when(scheduleEventRepos.findByName(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME))
        .thenReturn(ScheduleEventData.newTestInstance());
    controller.add(report);
  }

  @Test(expected = NotFoundException.class)
  public void testAddNoScheduleEvent() {
    report.setDevice(DeviceData.TEST_NAME);
    report.setEvent(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME);
    when(repos.save(report)).thenReturn(report);
    when(deviceRepos.findByName(DeviceData.TEST_NAME)).thenReturn(DeviceData.newTestInstance());
    when(scheduleEventRepos.findByName(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME))
        .thenReturn(null);
    assertEquals("Device Report ID returned is not the value expected", TEST_ID,
        controller.add(report));
  }

  @Test(expected = ServiceException.class)
  public void testAddWithNull() {
    controller.add(null);
  }

  @Test(expected = DataValidationException.class)
  public void testAddDuplicateKey() {
    report.setDevice(DeviceData.TEST_NAME);
    report.setEvent(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME);
    when(repos.save(report)).thenThrow(new DuplicateKeyException(TEST_ERR_MSG));
    when(deviceRepos.findByName(DeviceData.TEST_NAME)).thenReturn(DeviceData.newTestInstance());
    when(scheduleEventRepos.findByName(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME))
        .thenReturn(ScheduleEventData.newTestInstance());
    controller.add(report);
  }

  @Test(expected = ServiceException.class)
  public void testAddServiceException() {
    report.setDevice(DeviceData.TEST_NAME);
    report.setEvent(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME);
    when(repos.save(report)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    when(deviceRepos.findByName(DeviceData.TEST_NAME)).thenReturn(DeviceData.newTestInstance());
    when(scheduleEventRepos.findByName(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME))
        .thenReturn(ScheduleEventData.newTestInstance());
    controller.add(report);
  }

  @Test
  public void testUpdate() {
    report.setDevice(DeviceData.TEST_NAME);
    report.setEvent(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME);
    when(deviceRepos.findByName(DeviceData.TEST_NAME)).thenReturn(DeviceData.newTestInstance());
    when(scheduleEventRepos.findByName(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME))
        .thenReturn(ScheduleEventData.newTestInstance());
    when(dao.getByIdOrName(report)).thenReturn(report);
    assertTrue("Device Report was not updated", controller.update(report));
  }

  @Test
  public void testUpdateWithNoReportID() {
    report.setId(null);
    report.setDevice(DeviceData.TEST_NAME);
    report.setEvent(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME);
    when(deviceRepos.findByName(DeviceData.TEST_NAME)).thenReturn(DeviceData.newTestInstance());
    when(scheduleEventRepos.findByName(ScheduleEventData.TEST_SCHEDULE_EVENT_NAME))
        .thenReturn(ScheduleEventData.newTestInstance());
    when(dao.getByIdOrName(report)).thenReturn(report);
    assertTrue("Device Report was not updated", controller.update(report));
  }

  @Test(expected = ServiceException.class)
  public void testUpdateWithNull() {
    controller.update(null);
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateWithUnknownDeviceReport() {
    when(repos.findOne(TEST_ID)).thenReturn(null);
    controller.update(report);
  }

  @Test(expected = ServiceException.class)
  public void testUpdatException() {
    when(dao.getByIdOrName(report)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.update(report);
  }



  @Test
  public void testDelete() {
    when(repos.findOne(TEST_ID)).thenReturn(report);
    assertTrue("Device Report was not deleted", controller.delete(TEST_ID));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteNotFound() {
    when(repos.findOne(TEST_ID)).thenReturn(null);
    controller.delete(TEST_ID);
  }

  @Test(expected = ServiceException.class)
  public void testDeleteDaoFails() {
    when(repos.findOne(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.delete(TEST_ID);
  }

  @Test
  public void testDeleteByName() {
    when(repos.findByName(ReportData.TEST_RPT_NAME)).thenReturn(report);
    assertTrue("Device Report was not deleted", controller.deleteByName(ReportData.TEST_RPT_NAME));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteByNameNotFound() {
    when(repos.findByName(ReportData.TEST_RPT_NAME)).thenReturn(null);
    controller.deleteByName(ReportData.TEST_RPT_NAME);
  }

  @Test(expected = ServiceException.class)
  public void testDeleteByNameDaoFails() {
    when(repos.findByName(ReportData.TEST_RPT_NAME)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.deleteByName(ReportData.TEST_RPT_NAME);
  }

  private void setControllerMAXLIMIT(int newLimit) throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField(LIMIT_PROPERTY);
    temp.setAccessible(true);
    temp.set(controller, newLimit);
  }

}
